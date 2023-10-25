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
    data.append(FP16x16 { mag: 74549, sign: false });
    data.append(FP16x16 { mag: 86964, sign: false });
    data.append(FP16x16 { mag: 126384, sign: false });
    data.append(FP16x16 { mag: 109608, sign: false });
    data.append(FP16x16 { mag: 68987, sign: false });
    data.append(FP16x16 { mag: 81207, sign: false });
    data.append(FP16x16 { mag: 106493, sign: false });
    data.append(FP16x16 { mag: 89890, sign: false });
    data.append(FP16x16 { mag: 48135, sign: false });
    data.append(FP16x16 { mag: 55429, sign: false });
    data.append(FP16x16 { mag: 76585, sign: false });
    data.append(FP16x16 { mag: 57394, sign: false });
    TensorTrait::new(shape.span(), data.span())
}