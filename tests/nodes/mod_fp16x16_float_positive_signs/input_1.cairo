use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 350626, sign: false });
    data.append(FP16x16 { mag: 652090, sign: false });
    data.append(FP16x16 { mag: 425191, sign: false });
    data.append(FP16x16 { mag: 599148, sign: false });
    data.append(FP16x16 { mag: 74349, sign: false });
    data.append(FP16x16 { mag: 158675, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
