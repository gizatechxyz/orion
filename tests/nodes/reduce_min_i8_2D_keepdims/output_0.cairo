use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 0, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
