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

pub fn ramer_douglas_peucker(points: Vec<(f64, f64)>, epsilon: f64) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    if points.len() > 0 && epsilon >= 0.0 {
        result.push(points[0]);
        rdp(&points, epsilon, &mut result);
    }
    result
}
