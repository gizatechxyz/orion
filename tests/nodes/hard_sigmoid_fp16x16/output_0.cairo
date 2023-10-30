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
    data.append(FP16x16 { mag: 25594, sign: false });
    data.append(FP16x16 { mag: 63183, sign: false });
    data.append(FP16x16 { mag: 29206, sign: false });
    data.append(FP16x16 { mag: 51508, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
