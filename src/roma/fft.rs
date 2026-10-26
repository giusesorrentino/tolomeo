use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

pub struct FFTPlans {
    forward_plan: Arc<dyn RealToComplex<f64>>,
    backward_plan: Arc<dyn ComplexToReal<f64>>,
}

impl FFTPlans {
    pub fn new(len: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();

        Self {
            forward_plan: planner.plan_fft_forward(len),
            backward_plan: planner.plan_fft_inverse(len),
        }
    }

    pub fn workspace(&self) -> FFTWorkspace {
        FFTWorkspace {
            forward_plan: Arc::clone(&self.forward_plan),
            backward_plan: Arc::clone(&self.backward_plan),
            scratch_fwd: self.forward_plan.make_scratch_vec(),
            scratch_bwd: self.backward_plan.make_scratch_vec(),
        }
    }
}

pub struct FFTWorkspace {
    forward_plan: Arc<dyn RealToComplex<f64>>,
    backward_plan: Arc<dyn ComplexToReal<f64>>,
    scratch_fwd: Vec<Complex<f64>>,
    scratch_bwd: Vec<Complex<f64>>,
}

impl FFTWorkspace {
    pub fn forward(&mut self, input: &mut [f64], output: &mut [Complex<f64>]) {
        self.forward_plan
            .process_with_scratch(input, output, &mut self.scratch_fwd)
            .unwrap();
    }

    pub fn backward(&mut self, input: &mut [Complex<f64>], output: &mut [f64]) {
        self.backward_plan
            .process_with_scratch(input, output, &mut self.scratch_bwd)
            .unwrap();
    }
}
