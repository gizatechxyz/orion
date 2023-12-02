use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 131840, sign: false });
    data.append(FP16x16 { mag: 177152, sign: false });
    data.append(FP16x16 { mag: 8000, sign: true });
    data.append(FP16x16 { mag: 196224, sign: false });
    data.append(FP16x16 { mag: 92224, sign: true });
    data.append(FP16x16 { mag: 51296, sign: false });
    data.append(FP16x16 { mag: 106432, sign: false });
    data.append(FP16x16 { mag: 174720, sign: false });
    data.append(FP16x16 { mag: 140928, sign: true });
    data.append(FP16x16 { mag: 11464, sign: true });
    data.append(FP16x16 { mag: 140672, sign: true });
    data.append(FP16x16 { mag: 176384, sign: false });
    data.append(FP16x16 { mag: 104512, sign: false });
    data.append(FP16x16 { mag: 127168, sign: false });
    data.append(FP16x16 { mag: 7148, sign: true });
    data.append(FP16x16 { mag: 16544, sign: true });
    data.append(FP16x16 { mag: 60128, sign: false });
    data.append(FP16x16 { mag: 192512, sign: true });
    data.append(FP16x16 { mag: 152448, sign: true });
    data.append(FP16x16 { mag: 118208, sign: true });
    data.append(FP16x16 { mag: 55104, sign: true });
    data.append(FP16x16 { mag: 174592, sign: true });
    data.append(FP16x16 { mag: 49760, sign: true });
    data.append(FP16x16 { mag: 36192, sign: true });
    data.append(FP16x16 { mag: 26672, sign: false });
    data.append(FP16x16 { mag: 89984, sign: true });
    data.append(FP16x16 { mag: 112640, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
