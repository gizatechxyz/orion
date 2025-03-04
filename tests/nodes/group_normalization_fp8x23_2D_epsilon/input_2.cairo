use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1414509, sign: true });
    data.append(FP8x23 { mag: 1615431, sign: true });
    data.append(FP8x23 { mag: 9007479, sign: true });
    data.append(FP8x23 { mag: 7515349, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
