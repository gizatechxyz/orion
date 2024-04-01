use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(0);
    shape.append(8);
    shape.append(12);
    shape.append(15);

    let mut data = ArrayTrait::new();
    TensorTrait::new(shape.span(), data.span())
}
