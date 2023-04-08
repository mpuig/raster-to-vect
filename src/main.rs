use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;

use exoquant::{Color, convert_to_indexed, ditherer, optimizer};
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel, Rgb, RgbImage};
use image::buffer::ConvertBuffer;
use image::DynamicImage::ImageLuma8;
use imageproc::distance_transform::Norm;
use imageproc::drawing::Canvas;
use imageproc::edges::canny;
use imageproc::filter::{gaussian_blur_f32, median_filter};
use imageproc::morphology::{close, dilate, erode};
use svg::Document;
use svg::node::element::{Path as SvgPath, Rectangle};
use svg::node::element::path::Data;

use structs::{Path, Point, Segment};

mod rdp;
mod structs;


fn main() {
    let num_colors = 8;
    let gaussian_blur_dev = 1.5;
    let edge_low_threshold = 10.0;
    let edge_high_threshold = 50.0;

    println!("Read the input bitmap image.");
    let color_image = image::open("input.png").unwrap();
    let dimensions = GenericImageView::dimensions(&color_image);

    println!("Quantize the image using k-means clustering");
    let quantized_image = quantize_colors(&color_image, num_colors);
    let filtered_quantized_image = median_filter(&quantized_image, 1, 1);
    if let Err(e) = filtered_quantized_image.save_with_format("1_quantized_image.png", image::ImageFormat::Png) {
        eprintln!("Error saving edge image: {:?}", e);
    }

    println!("Preprocess the image");
    let edge_image = edge_detection(&filtered_quantized_image, gaussian_blur_dev, edge_low_threshold, edge_high_threshold);

    println!("Extract contours from the edge-detected image");
    let contours = extract_contours(&edge_image, &filtered_quantized_image);

    println!("Apply the tracing algorithm");
    let vector_data = trace_bitmap(contours, &filtered_quantized_image);

    println!("Export the vector data to a file format (e.g., SVG)");
    match export_vector_data(&vector_data, "output.svg", dimensions.0, dimensions.1) {
        Ok(_) => println!("Vector data successfully exported."),
        Err(e) => eprintln!("Failed to export vector data: {}", e),
    }
}

fn quantize_colors(image: &DynamicImage, num_colors: usize) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    // Apply Gaussian blur, adjust the sigma value to control the amount of blur
    //let blurred_image = image::imageops::blur(image, 4.5);

    let pixels = image
        .pixels()
        .map(|(_, _, p)| {
            let cols = p.channels();
            Color::new(cols[0], cols[1], cols[2], cols[3])
        })
        .collect::<Vec<_>>();
    let width = image.width() as usize;
    let height = image.height() as usize;
    let (palette, indexed_pixels) = convert_to_indexed(
        &pixels,
        width,
        num_colors,
        &optimizer::KMeans,
        &ditherer::FloydSteinberg::new(),
    );
    // Create the final image by color lookup from the palette
    let output_data: Vec<u8> = indexed_pixels
        .iter()
        .flat_map(|&color_index| {
            let color = palette.get(color_index as usize).unwrap();
            vec![color.r, color.g, color.b]
        })
        .collect();

    RgbImage::from_raw(width as u32, height as u32, output_data).unwrap()
}

fn edge_detection(color_image: &RgbImage, gaussian_blur_dev: f32, edge_low_threshold: f32, edge_high_threshold: f32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    println!("Convert the image to grayscale");
    let grayscale_image: GrayImage = color_image.convert();
    if let Err(e) = grayscale_image.save_with_format("3_grayscale_image.png", image::ImageFormat::Png) {
        eprintln!("Error saving edge image: {:?}", e);
    }
    let filtered_image = median_filter(&grayscale_image, 1, 1);
    if let Err(e) = filtered_image.save_with_format("3_filtered_image.png", image::ImageFormat::Png) {
        eprintln!("Error saving edge image: {:?}", e);
    }

    println!("Apply the Canny edge detection algorithm");
    let edge_image = canny(&filtered_image, edge_low_threshold, edge_high_threshold);
    if let Err(e) = edge_image.save_with_format("3_edge_image.png", image::ImageFormat::Png) {
        eprintln!("Error saving edge image: {:?}", e);
    }


    edge_image
}

fn trace_bitmap(contours: Vec<Path>, color_image: &RgbImage) -> Vec<Path> {
    println!("Simplify the contours using the Ramer-Douglas-Peucker algorithm");
    let epsilon = 1.0;
    let simplified_contours: Vec<Path> = contours
        .into_iter()
        .map(|path| {
            let points: Vec<(f64, f64)> = path
                .points_along_path(1)
                .into_iter()
                .map(|point| (point.x, point.y))
                .collect();

            let simplified_points = rdp::ramer_douglas_peucker(points, epsilon);

            let mut simplified_path = Path::new(Point::new(simplified_points[0].0, simplified_points[0].1), path.color);
            for point in simplified_points.iter().skip(1) {
                simplified_path.line_to(Point::new(point.0, point.1));
            }
            simplified_path
        })
        .collect();

    simplified_contours
}

fn extract_contours(edge_image: &GrayImage, color_image: &RgbImage) -> Vec<Path> {
    let mut contours = Vec::new();
    let mut visited = vec![vec![false; edge_image.height() as usize]; edge_image.width() as usize];

    for (x, y, pixel) in edge_image.enumerate_pixels() {
        if pixel == &Luma([255u8]) && !visited[x as usize][y as usize] {
            let contour_points = trace_contour(edge_image, color_image, &mut visited, x, y);
            if !contour_points.is_empty() {
                let mut contour_path = Path::new(contour_points[0].0, contour_points[0].1);
                for (point, color) in contour_points.iter().skip(1) {
                    contour_path.line_to_with_color(*point, *color);
                }
                contours.push(contour_path);
            }
        }
    }
    contours
}

fn trace_contour(edge_image: &GrayImage, color_image: &RgbImage, visited: &mut Vec<Vec<bool>>, x: u32, y: u32) -> Vec<(Point, Rgb<u8>)> {
    let width = edge_image.width();
    let height = edge_image.height();

    let is_inside = |x: i32, y: i32| x >= 0 && y >= 0 && x < width as i32 && y < height as i32;
    let neighbors: [(i32, i32); 8] = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)];

    let mut contour_points = Vec::new();
    let mut stack = VecDeque::new();
    stack.push_back((x as i32, y as i32));

    while let Some(current_point) = stack.pop_front() {
        let x = current_point.0;
        let y = current_point.1;

        if is_inside(x, y) && !visited[x as usize][y as usize] && edge_image.get_pixel(x as u32, y as u32).0[0] != 0 {
            visited[x as usize][y as usize] = true;
            contour_points.push((
                Point::new(current_point.0 as f64, current_point.1 as f64),
                color_image.get_pixel(current_point.0 as u32, current_point.1 as u32).clone(),
            ));
            for &(dx, dy) in &neighbors {
                let new_x = x + dx;
                let new_y = y + dy;

                if is_inside(new_x, new_y) && !visited[new_x as usize][new_y as usize] {
                    stack.push_back((new_x, new_y));
                }
            }
        }
    }
    contour_points
}

fn export_vector_data(paths: &[Path], file_name: &str, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
    let mut document = Document::new()
        .set("viewBox", (0, 0, width, height))
        .set("width", width)
        .set("height", height);

    // Add a white background rectangle
    let background = Rectangle::new()
        .set("width", "100%")
        .set("height", "100%")
        .set("fill", "white");
    document = document.add(background);

    // Add traced paths to the SVG document
    for path in paths {
        let mut data = Data::new().move_to((path.start.x, path.start.y));

        for segment in &path.segments {
            match segment {
                Segment::Line(end, ..) => {
                    data = data.line_to((end.x, end.y));
                }
                Segment::QuadraticBezier(control, end) => {
                    data = data.quadratic_curve_to(((control.x, control.y), (end.x, end.y)));
                }
            }
        }
        let red = path.color[0] as u8;
        let green = path.color[1] as u8;
        let blue = path.color[2] as u8;
        let color_string = format!("rgb({}, {}, {})", red, green, blue);


        let svg_path = SvgPath::new()
            .set("fill", "none")
            .set("stroke", color_string)
            .set("stroke-width", 1)
            .set("d", data);

        document = document.add(svg_path);
    }

    // Write the SVG data to a file
    let mut file = File::create(file_name)?;
    file.write_all(document.to_string().as_bytes())?;

    Ok(())
}
