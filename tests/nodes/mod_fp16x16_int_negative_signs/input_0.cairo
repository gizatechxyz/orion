use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 458752, sign: true });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 524288, sign: true });
    data.append(FP16x16 { mag: 589824, sign: true });
    data.append(FP16x16 { mag: 196608, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
