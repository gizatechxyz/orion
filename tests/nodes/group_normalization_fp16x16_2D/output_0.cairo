use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 38546, sign: false });
    data.append(FP16x16 { mag: 16268, sign: false });
    data.append(FP16x16 { mag: 62831, sign: false });
    data.append(FP16x16 { mag: 61214, sign: false });
    data.append(FP16x16 { mag: 38546, sign: false });
    data.append(FP16x16 { mag: 16268, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
