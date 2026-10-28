pub trait LinearOperator {
    fn project(&mut self, x: &[f64], out: &mut [f64]);
    fn precondition(&mut self, x: &[f64], out: &mut [f64]);
}
