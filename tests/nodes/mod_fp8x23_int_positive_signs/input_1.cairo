use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 83886080, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: false });
    data.append(FP8x23 { mag: 8388608, sign: false });
    data.append(FP8x23 { mag: 75497472, sign: false });
    data.append(FP8x23 { mag: 8388608, sign: false });
    data.append(FP8x23 { mag: 58720256, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
