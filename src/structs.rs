use image::Rgb;

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
