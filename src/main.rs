use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;

use exoquant::{Color, convert_to_indexed, ditherer, optimizer};
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel, Rgb, RgbImage};
use imageproc::edges::canny;
use imageproc::filter::gaussian_blur_f32;
use svg::Document;
use svg::node::element::{Path as SvgPath, Rectangle};
use svg::node::element::path::Data;

fn main() {
    println!("Read the input bitmap image.");
    let color_image = image::open("input.png").unwrap();

    println!("Quantize the image using k-means clustering");
    let num_colors = 16; // Adjust this value based on your requirements
    // let quantized_image = quantize_colors(&color_image.to_rgb8(), num_colors);
    let quantized_image = encode(&color_image, num_colors);

    println!("Preprocess the image (optional)");
    let preprocessed_image = preprocess_image(&color_image);

    println!("Apply the tracing algorithm");
    let vector_data = trace_bitmap(&preprocessed_image, &quantized_image);

    println!("Export the vector data to a file format (e.g., SVG)");
    match export_vector_data(&vector_data, "output.svg", color_image.width(), color_image.height()) {
        Ok(_) => println!("Vector data successfully exported."),
        Err(e) => eprintln!("Failed to export vector data: {}", e),
    }
}

fn encode(imag: &image::DynamicImage, num_colors: usize) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let pixels = imag
        .pixels()
        .map(|(_, _, p)| {
            let cols = p.channels();
            Color::new(cols[0], cols[1], cols[2], cols[3])
        })
        .collect::<Vec<_>>();
    let width = imag.width() as usize;
    let height = imag.height() as usize;
    let (_, indexed_pixels) = convert_to_indexed(
        &pixels,
        width,
        num_colors,
        &optimizer::KMeans,
        &ditherer::FloydSteinberg::checkered(),
    );

    // Allocate a new buffer for the RGB image, 3 bytes per pixel
    let mut output_data = vec![0u8; width * height * 3];

    let mut i = 0;
    // Iterate through 4-byte chunks of the image data (RGBA bytes)
    for chunk in indexed_pixels.chunks(4) {
        // ... and copy each of them to output, leaving out the A byte
        output_data[i..i + 3].copy_from_slice(&chunk[0..3]);
        i += 3;
    }
    RgbImage::from_raw(width as u32, height as u32, output_data).unwrap()
}

fn preprocess_image(image: &DynamicImage) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    // Convert the image to grayscale
    let grayscale_image = image.to_luma8();

    // Apply Gaussian blur with a specified standard deviation
    let std_dev = 1.5; // Adjust this value based on your requirements
    let blurred_image = gaussian_blur_f32(&grayscale_image, std_dev);

    blurred_image
}

// Returns the distance from point p to the line between p1 and p2
fn perpendicular_distance(p: &(f64, f64), p1: &(f64, f64), p2: &(f64, f64)) -> f64 {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    (p.0 * dy - p.1 * dx + p2.0 * p1.1 - p2.1 * p1.0).abs() / dx.hypot(dy)
}

fn rdp(points: &[(f64, f64)], epsilon: f64, result: &mut Vec<(f64, f64)>) {
    let n = points.len();
    if n < 2 {
        return;
    }
    let mut max_dist = 0.0;
    let mut index = 0;
    for i in 1..n - 1 {
        let dist = perpendicular_distance(&points[i], &points[0], &points[n - 1]);
        if dist > max_dist {
            max_dist = dist;
            index = i;
        }
    }
    if max_dist > epsilon {
        rdp(&points[0..=index], epsilon, result);
        rdp(&points[index..n], epsilon, result);
    } else {
        result.push(points[n - 1]);
    }
}

fn ramer_douglas_peucker(points: Vec<(f64, f64)>, epsilon: f64) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    if points.len() > 0 && epsilon >= 0.0 {
        result.push(points[0]);
        rdp(&points, epsilon, &mut result);
    }
    result
}

fn trace_bitmap(image: &ImageBuffer<Luma<u8>, Vec<u8>>, color_image: &RgbImage) -> Vec<Path> {
    println!("Apply the Canny edge detection algorithm");
    let low_threshold = 10.0;
    let high_threshold = 50.0;
    let edge_image = canny(image, low_threshold, high_threshold);

    println!("Extract contours from the edge-detected image");
    let contours = extract_contours(&edge_image, color_image);
    println!(" -> Extracted contours: {}", contours.len());

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

            let simplified_points = ramer_douglas_peucker(points, epsilon);

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

        let svg_path = SvgPath::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 1)
            .set("d", data);

        document = document.add(svg_path);
    }

    // Write the SVG data to a file
    let mut file = File::create(file_name)?;
    file.write_all(document.to_string().as_bytes())?;

    Ok(())
}

// First, let's create a Point struct to represent 2D points:
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

// Next, we'll create a Segment enum to represent line segments and quadratic Bezier curves:
pub enum Segment {
    Line(Point, Rgb<u8>),
    QuadraticBezier(Point, Point),
}

// Now, we can define the Path struct and implement methods for adding
// lines and curves, as well as a method for calculating the points along the path.
// The Path struct could be represented by a collection of points and Bezier curves
pub struct Path {
    pub start: Point,
    pub color: Rgb<u8>,
    pub segments: Vec<Segment>,
}

impl Path {
    pub fn new(start: Point, color: Rgb<u8>) -> Self {
        Self {
            start,
            color,
            segments: Vec::new(),
        }
    }

    pub fn line_to(&mut self, end: Point) {
        self.segments.push(Segment::Line(end, self.color));
    }

    pub fn line_to_with_color(&mut self, end: Point, color: Rgb<u8>) {
        self.segments.push(Segment::Line(end, color));
    }

    pub fn quadratic_bezier_to(&mut self, control: Point, end: Point) {
        self.segments.push(Segment::QuadraticBezier(control, end));
    }

    pub fn points_along_path(&self, resolution: usize) -> Vec<Point> {
        let mut points = vec![self.start];
        for segment in &self.segments {
            match segment {
                Segment::Line(end, ..) => {
                    points.push(*end);
                }
                Segment::QuadraticBezier(control, end) => {
                    for i in 1..=resolution {
                        let t = i as f64 / resolution as f64;
                        let point = quadratic_bezier(self.start, *control, *end, t);
                        points.push(point);
                    }
                }
            }
        }
        points
    }
}

fn quadratic_bezier(p0: Point, p1: Point, p2: Point, t: f64) -> Point {
    let x = (1.0 - t).powi(2) * p0.x + 2.0 * (1.0 - t) * t * p1.x + t.powi(2) * p2.x;
    let y = (1.0 - t).powi(2) * p0.y + 2.0 * (1.0 - t) * t * p1.y + t.powi(2) * p2.y;
    Point::new(x, y)
}
