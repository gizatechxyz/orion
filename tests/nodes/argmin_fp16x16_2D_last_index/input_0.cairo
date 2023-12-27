use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1376256, sign: true });
    data.append(FP16x16 { mag: 5898240, sign: true });
    data.append(FP16x16 { mag: 6356992, sign: true });
    data.append(FP16x16 { mag: 2883584, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
