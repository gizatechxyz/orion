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
    data.append(FP16x16 { mag: 146941, sign: true });
    data.append(FP16x16 { mag: 35814, sign: true });
    data.append(FP16x16 { mag: 157339, sign: false });
    data.append(FP16x16 { mag: 75115, sign: true });
    data.append(FP16x16 { mag: 153883, sign: false });
    data.append(FP16x16 { mag: 170137, sign: false });
    data.append(FP16x16 { mag: 163686, sign: true });
    data.append(FP16x16 { mag: 89744, sign: true });
    data.append(FP16x16 { mag: 59131, sign: true });
    data.append(FP16x16 { mag: 267, sign: true });
    data.append(FP16x16 { mag: 20112, sign: false });
    data.append(FP16x16 { mag: 3370, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
