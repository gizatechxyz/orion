use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1430064, sign: true });
    data.append(FP8x23 { mag: 15545622, sign: true });
    data.append(FP8x23 { mag: 469121, sign: true });
    data.append(FP8x23 { mag: 24424260, sign: true });
    TensorTrait::new(shape.span(), data.span())
}