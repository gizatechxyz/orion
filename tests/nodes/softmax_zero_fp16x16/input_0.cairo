use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2278, sign: true });
    data.append(FP16x16 { mag: 76483, sign: false });
    data.append(FP16x16 { mag: 23998, sign: false });
    data.append(FP16x16 { mag: 808, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
