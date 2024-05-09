use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 262415, sign: true });
    data.append(FP16x16 { mag: 144773, sign: true });
    data.append(FP16x16 { mag: 248695, sign: true });
    data.append(FP16x16 { mag: 22664, sign: true });
    data.append(FP16x16 { mag: 601861, sign: true });
    data.append(FP16x16 { mag: 50086, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
