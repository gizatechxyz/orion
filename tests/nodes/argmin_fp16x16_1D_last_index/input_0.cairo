use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7143424, sign: false });
    data.append(FP16x16 { mag: 5177344, sign: false });
    data.append(FP16x16 { mag: 5963776, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
