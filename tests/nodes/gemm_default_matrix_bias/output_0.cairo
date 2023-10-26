use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 150288, sign: false });
    data.append(FP16x16 { mag: 128258, sign: false });
    data.append(FP16x16 { mag: 118559, sign: false });
    data.append(FP16x16 { mag: 113724, sign: false });
    data.append(FP16x16 { mag: 135691, sign: false });
    data.append(FP16x16 { mag: 154786, sign: false });
    data.append(FP16x16 { mag: 152316, sign: false });
    data.append(FP16x16 { mag: 135919, sign: false });
    data.append(FP16x16 { mag: 168184, sign: false });
    data.append(FP16x16 { mag: 160656, sign: false });
    data.append(FP16x16 { mag: 162874, sign: false });
    data.append(FP16x16 { mag: 125446, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
