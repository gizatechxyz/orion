use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 22388, sign: false });
    data.append(FP16x16 { mag: 10521, sign: false });
    data.append(FP16x16 { mag: 51858, sign: false });
    data.append(FP16x16 { mag: 60920, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
