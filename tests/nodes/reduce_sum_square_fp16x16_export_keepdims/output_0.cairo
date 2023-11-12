use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5, sign: false });
    data.append(FP16x16 { mag: 25, sign: false });
    data.append(FP16x16 { mag: 61, sign: false });
    data.append(FP16x16 { mag: 113, sign: false });
    data.append(FP16x16 { mag: 181, sign: false });
    data.append(FP16x16 { mag: 265, sign: false });
    TensorTrait::new(shape.span(), data.span())
}