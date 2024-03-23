use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 8474, sign: false });
    data.append(FP16x16 { mag: 64617, sign: true });
    data.append(FP16x16 { mag: 47921, sign: false });
    data.append(FP16x16 { mag: 78006, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
