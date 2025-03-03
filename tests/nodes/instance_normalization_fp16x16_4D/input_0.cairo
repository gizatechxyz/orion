use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 21977, sign: false });
    data.append(FP16x16 { mag: 43336, sign: true });
    data.append(FP16x16 { mag: 37383, sign: false });
    data.append(FP16x16 { mag: 44745, sign: true });
    data.append(FP16x16 { mag: 107958, sign: false });
    data.append(FP16x16 { mag: 13941, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
