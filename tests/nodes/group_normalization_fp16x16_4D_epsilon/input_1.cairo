use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 92821, sign: true });
    data.append(FP16x16 { mag: 66307, sign: false });
    data.append(FP16x16 { mag: 55436, sign: false });
    data.append(FP16x16 { mag: 48308, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
