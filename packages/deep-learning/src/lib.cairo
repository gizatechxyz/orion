pub mod tensor;

use tensor::{Tensor, TensorAdd};

use orion_numbers::{F64, F64Impl};


fn main(x: Tensor<F64>, y: Tensor<F64>) -> Tensor<F64> {
    x + y
}
