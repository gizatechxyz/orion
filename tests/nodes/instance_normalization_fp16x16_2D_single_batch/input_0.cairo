use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 99266, sign: false });
    data.append(FP16x16 { mag: 2539, sign: false });
    data.append(FP16x16 { mag: 27765, sign: true });
    data.append(FP16x16 { mag: 106225, sign: true });
    data.append(FP16x16 { mag: 83195, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
