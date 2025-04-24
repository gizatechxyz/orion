use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 67631, sign: false });
    data.append(FP16x16 { mag: 8091, sign: true });
    data.append(FP16x16 { mag: 146876, sign: false });
    data.append(FP16x16 { mag: 20846, sign: true });
    data.append(FP16x16 { mag: 22875, sign: true });
    data.append(FP16x16 { mag: 71324, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
