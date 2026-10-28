use rayon::prelude::*;

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x * y).sum()
}
