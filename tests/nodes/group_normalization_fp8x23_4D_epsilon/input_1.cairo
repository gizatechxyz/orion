use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 6635423, sign: true });
    data.append(FP8x23 { mag: 5563348, sign: true });
    data.append(FP8x23 { mag: 3043715, sign: false });
    data.append(FP8x23 { mag: 11763010, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
