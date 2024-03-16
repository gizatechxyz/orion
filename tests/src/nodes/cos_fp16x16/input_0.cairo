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
    data.append(FP16x16 { mag: 1737975, sign: false });
    data.append(FP16x16 { mag: 6312706, sign: false });
    data.append(FP16x16 { mag: 5651744, sign: false });
    data.append(FP16x16 { mag: 6320564, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
