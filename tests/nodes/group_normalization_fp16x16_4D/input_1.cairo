use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 85417, sign: true });
    data.append(FP16x16 { mag: 24039, sign: true });
    data.append(FP16x16 { mag: 61953, sign: true });
    data.append(FP16x16 { mag: 9920, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
