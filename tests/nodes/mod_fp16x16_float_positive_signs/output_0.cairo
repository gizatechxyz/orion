use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 70366, sign: false });
    data.append(FP16x16 { mag: 41219, sign: false });
    data.append(FP16x16 { mag: 220709, sign: false });
    data.append(FP16x16 { mag: 331174, sign: false });
    data.append(FP16x16 { mag: 10860, sign: false });
    data.append(FP16x16 { mag: 37015, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
