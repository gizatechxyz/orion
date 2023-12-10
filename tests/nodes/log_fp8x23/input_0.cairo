use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 193551844, sign: false });
    data.append(FP8x23 { mag: 202689108, sign: false });
    data.append(FP8x23 { mag: 535508695, sign: false });
    data.append(FP8x23 { mag: 235726980, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
