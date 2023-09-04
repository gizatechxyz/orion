use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 401540, sign: false });
    data.append(FP16x16 { mag: 47813, sign: true });
    data.append(FP16x16 { mag: 206992, sign: false });
    data.append(FP16x16 { mag: 323041, sign: true });
    TensorTrait::new(shape.span(), data.span())
}