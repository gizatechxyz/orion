use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 578813952, sign: false });
    data.append(FP8x23 { mag: 771751936, sign: true });
    data.append(FP8x23 { mag: 394264576, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
