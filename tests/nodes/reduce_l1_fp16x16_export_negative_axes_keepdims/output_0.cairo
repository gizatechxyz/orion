use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 30, sign: false });
    data.append(FP16x16 { mag: 33, sign: false });
    data.append(FP16x16 { mag: 36, sign: false });
    data.append(FP16x16 { mag: 39, sign: false });
    data.append(FP16x16 { mag: 42, sign: false });
    data.append(FP16x16 { mag: 45, sign: false });
    data.append(FP16x16 { mag: 48, sign: false });
    data.append(FP16x16 { mag: 51, sign: false });
    data.append(FP16x16 { mag: 54, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
