use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 78508, sign: true });
    data.append(FP16x16 { mag: 16306, sign: true });
    data.append(FP16x16 { mag: 110399, sign: false });
    data.append(FP16x16 { mag: 25434, sign: false });
    data.append(FP16x16 { mag: 8759, sign: true });
    data.append(FP16x16 { mag: 61322, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
