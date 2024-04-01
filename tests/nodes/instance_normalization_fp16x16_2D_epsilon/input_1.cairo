use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 49720, sign: false });
    data.append(FP16x16 { mag: 130140, sign: true });
    data.append(FP16x16 { mag: 4590, sign: true });
    data.append(FP16x16 { mag: 86674, sign: false });
    data.append(FP16x16 { mag: 27027, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
