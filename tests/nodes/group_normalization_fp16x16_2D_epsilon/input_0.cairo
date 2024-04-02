use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 9593, sign: false });
    data.append(FP16x16 { mag: 96795, sign: false });
    data.append(FP16x16 { mag: 164084, sign: false });
    data.append(FP16x16 { mag: 5388, sign: false });
    data.append(FP16x16 { mag: 108895, sign: false });
    data.append(FP16x16 { mag: 8978, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
