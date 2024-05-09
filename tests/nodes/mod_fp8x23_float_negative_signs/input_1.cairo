use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 83597096, sign: true });
    data.append(FP8x23 { mag: 83122912, sign: true });
    data.append(FP8x23 { mag: 42025356, sign: true });
    data.append(FP8x23 { mag: 58904400, sign: true });
    data.append(FP8x23 { mag: 54996400, sign: true });
    data.append(FP8x23 { mag: 83000088, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
