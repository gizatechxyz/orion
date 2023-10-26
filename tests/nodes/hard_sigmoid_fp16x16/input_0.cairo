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
    data.append(FP16x16 { mag: 35866, sign: true });
    data.append(FP16x16 { mag: 152077, sign: false });
    data.append(FP16x16 { mag: 17807, sign: true });
    data.append(FP16x16 { mag: 93701, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
