use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5029, sign: true });
    data.append(FP16x16 { mag: 37483, sign: false });
    data.append(FP16x16 { mag: 74737, sign: false });
    data.append(FP16x16 { mag: 47732, sign: false });
    data.append(FP16x16 { mag: 120075, sign: true });
    data.append(FP16x16 { mag: 21966, sign: false });
    data.append(FP16x16 { mag: 25026, sign: true });
    data.append(FP16x16 { mag: 35991, sign: true });
    data.append(FP16x16 { mag: 35005, sign: false });
    data.append(FP16x16 { mag: 41534, sign: false });
    data.append(FP16x16 { mag: 28951, sign: false });
    data.append(FP16x16 { mag: 9321, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
