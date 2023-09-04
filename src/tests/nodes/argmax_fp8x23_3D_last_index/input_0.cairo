use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 645922816, sign: false });
    data.append(FP8x23 { mag: 335544320, sign: true });
    data.append(FP8x23 { mag: 889192448, sign: false });
    data.append(FP8x23 { mag: 813694976, sign: true });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 645922816, sign: false });
    data.append(FP8x23 { mag: 981467136, sign: true });
    data.append(FP8x23 { mag: 536870912, sign: true });
    TensorTrait::new(shape.span(), data.span())
}