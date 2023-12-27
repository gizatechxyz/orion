use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 654311424, sign: true });
    data.append(FP8x23 { mag: 469762048, sign: false });
    data.append(FP8x23 { mag: 964689920, sign: true });
    data.append(FP8x23 { mag: 662700032, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
