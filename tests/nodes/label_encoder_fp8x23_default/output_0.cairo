use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(11);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 92274688, sign: false });
    data.append(FP8x23 { mag: 184549376, sign: false });
    data.append(FP8x23 { mag: 830472192, sign: false });
    data.append(FP8x23 { mag: 830472192, sign: false });
    data.append(FP8x23 { mag: 461373440, sign: false });
    data.append(FP8x23 { mag: 553648128, sign: false });
    data.append(FP8x23 { mag: 92274688, sign: false });
    data.append(FP8x23 { mag: 184549376, sign: false });
    data.append(FP8x23 { mag: 830472192, sign: false });
    data.append(FP8x23 { mag: 645922816, sign: false });
    data.append(FP8x23 { mag: 830472192, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
