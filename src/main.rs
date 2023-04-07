use image::{DynamicImage, ImageBuffer, Luma, GrayImage};
use imageproc::filter::{gaussian_blur_f32};
use imageproc::edges;
use std::fs::File;
use std::io::Write;
use svg::Document;
use svg::node::element::path::Data;
use svg::node::element::{Path as SvgPath, Rectangle};


fn main() {
    println!("Read the input bitmap image.");
    let input_image = image::open("input.png").unwrap();

    println!("Preprocess the image (optional)");
    let preprocessed_image = preprocess_image(&input_image);

    println!("Apply the tracing algorithm");
    let vector_data = trace_bitmap(&preprocessed_image);

    println!("Export the vector data to a file format (e.g., SVG)");
    match export_vector_data(&vector_data, "output.svg", input_image.width(), input_image.height()) {
        Ok(_) => println!("Vector data successfully exported."),
        Err(e) => eprintln!("Failed to export vector data: {}", e),
    }
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

fn trace_bitmap(image: &GrayImage) -> Vec<Path> {
    println!("Apply the Canny edge detection algorithm");
    let low_threshold = 10.0;
    let high_threshold = 50.0;
    let canny_image = edges::canny(image, low_threshold, high_threshold);

    println!("Extract contours from the edge-detected image");
    let contours = extract_contours(&canny_image);

    println!("Simplify the contours using the Ramer-Douglas-Peucker algorithm");
    let epsilon = 0.1;
    let simplified_contours: Vec<Path> = contours
        .into_iter()
        .map(|path| {
            let points: Vec<(f64, f64)> = path
                .points_along_path(1)
                .into_iter()
                .map(|point| (point.x, point.y))
                .collect();

            let simplified_points = ramer_douglas_peucker(points, epsilon);

            let mut simplified_path = Path::new(Point::new(simplified_points[0].0, simplified_points[0].1));
            for point in simplified_points.iter().skip(1) {
                simplified_path.line_to(Point::new(point.0, point.1));
            }
            simplified_path
        })
        .collect();

    simplified_contours
}

fn extract_contours(image: &GrayImage) -> Vec<Path> {
    let mut contours = Vec::new();
    let mut visited = vec![vec![false; image.height() as usize]; image.width() as usize];

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel == &Luma([255u8]) && !visited[x as usize][y as usize] {
            let contour_points = moore_neighbor_tracing(image, &mut visited, x, y);
            if !contour_points.is_empty() {
                let mut contour_path = Path::new(contour_points[0]);
                for point in contour_points.iter().skip(1) {
                    contour_path.line_to(*point);
                }
                contours.push(contour_path);
            }
        }
    }

    contours
}

fn moore_neighbor_tracing(image: &GrayImage, visited: &mut Vec<Vec<bool>>, x: u32, y: u32) -> Vec<Point> {
    let mut contour_points = Vec::new();
    let mut current_point = (x, y);
    let mut backtrack_point = (0, 0);
    let mut backtracked = false;

    let moore_neighbors = [
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
    ];

    while !visited[current_point.0 as usize][current_point.1 as usize] || !backtracked {
        visited[current_point.0 as usize][current_point.1 as usize] = true;
        contour_points.push(Point::new(current_point.0 as f64, current_point.1 as f64));

        let start_index = if backtracked {
            (backtrack_point.0 + 1) % 8
        } else {
            0
        };

        backtracked = false;

        for i in start_index..8 {
            let next_point = (
                current_point.0 as i32 + moore_neighbors[i].0,
                current_point.1 as i32 + moore_neighbors[i].1,
            );

            if next_point.0 >= 0
                && next_point.1 >= 0
                && next_point.0 < image.width() as i32
                && next_point.1 < image.height() as i32
                && image.get_pixel(next_point.0 as u32, next_point.1 as u32) == &Luma([255u8])
                && !visited[next_point.0 as usize][next_point.1 as usize]
            {
                backtrack_point = (i, i);
                current_point = (next_point.0 as u32, next_point.1 as u32);
                break;
            } else if i == 7 {
                backtracked = true;
                backtrack_point = (backtrack_point.1, (backtrack_point.1 + 1) % 8);
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
                Segment::Line(end) => {
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
    Line(Point),
    QuadraticBezier(Point, Point),
}

// Now, we can define the Path struct and implement methods for adding
// lines and curves, as well as a method for calculating the points along the path.
// The Path struct could be represented by a collection of points and Bezier curves
pub struct Path {
    pub start: Point,
    pub segments: Vec<Segment>,
}

impl Path {
    pub fn new(start: Point) -> Self {
        Self {
            start,
            segments: Vec::new(),
        }
    }

    pub fn line_to(&mut self, end: Point) {
        self.segments.push(Segment::Line(end));
    }

    pub fn quadratic_bezier_to(&mut self, control: Point, end: Point) {
        self.segments.push(Segment::QuadraticBezier(control, end));
    }

    pub fn points_along_path(&self, resolution: usize) -> Vec<Point> {
        let mut points = vec![self.start];
        for segment in &self.segments {
            match segment {
                Segment::Line(end) => {
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
