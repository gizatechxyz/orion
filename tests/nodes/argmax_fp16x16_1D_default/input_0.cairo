use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 4194304, sign: true });
    data.append(FP16x16 { mag: 917504, sign: false });
    data.append(FP16x16 { mag: 5832704, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
