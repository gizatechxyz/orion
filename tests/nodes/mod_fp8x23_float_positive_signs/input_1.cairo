use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 52038744, sign: false });
    data.append(FP8x23 { mag: 85696952, sign: false });
    data.append(FP8x23 { mag: 77523360, sign: false });
    data.append(FP8x23 { mag: 56766176, sign: false });
    data.append(FP8x23 { mag: 19128604, sign: false });
    data.append(FP8x23 { mag: 60584360, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
