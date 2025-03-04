use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2185473, sign: false });
    data.append(FP8x23 { mag: 3865267, sign: false });
    data.append(FP8x23 { mag: 9912936, sign: false });
    data.append(FP8x23 { mag: 5109914, sign: false });
    data.append(FP8x23 { mag: 7929377, sign: false });
    data.append(FP8x23 { mag: 331705, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
