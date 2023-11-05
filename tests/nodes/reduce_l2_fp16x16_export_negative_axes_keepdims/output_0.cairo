use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5, sign: false });
    data.append(FP16x16 { mag: 5, sign: false });
    data.append(FP16x16 { mag: 6, sign: false });
    data.append(FP16x16 { mag: 6, sign: false });
    data.append(FP16x16 { mag: 6, sign: false });
    data.append(FP16x16 { mag: 6, sign: false });
    data.append(FP16x16 { mag: 6, sign: false });
    data.append(FP16x16 { mag: 7, sign: false });
    data.append(FP16x16 { mag: 7, sign: false });
    TensorTrait::new(shape.span(), data.span())
}