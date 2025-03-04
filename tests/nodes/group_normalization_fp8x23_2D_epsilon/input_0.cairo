use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 251575, sign: false });
    data.append(FP8x23 { mag: 11263120, sign: true });
    data.append(FP8x23 { mag: 10489348, sign: false });
    data.append(FP8x23 { mag: 3873435, sign: false });
    data.append(FP8x23 { mag: 1827669, sign: false });
    data.append(FP8x23 { mag: 461198, sign: true });
    data.append(FP8x23 { mag: 15252800, sign: false });
    data.append(FP8x23 { mag: 2498850, sign: false });
    data.append(FP8x23 { mag: 10601261, sign: true });
    data.append(FP8x23 { mag: 17650742, sign: true });
    data.append(FP8x23 { mag: 20032782, sign: true });
    data.append(FP8x23 { mag: 3004365, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
