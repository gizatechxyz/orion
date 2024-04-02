use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 59552, sign: true });
    data.append(FP16x16 { mag: 49297, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
