use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(27);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 112064, sign: false });
    data.append(FP16x16 { mag: 56256, sign: false });
    data.append(FP16x16 { mag: 32528, sign: false });
    data.append(FP16x16 { mag: 22672, sign: false });
    data.append(FP16x16 { mag: 16752, sign: false });
    data.append(FP16x16 { mag: 71168, sign: false });
    data.append(FP16x16 { mag: 152704, sign: false });
    data.append(FP16x16 { mag: 148352, sign: false });
    data.append(FP16x16 { mag: 41536, sign: false });
    data.append(FP16x16 { mag: 42272, sign: false });
    data.append(FP16x16 { mag: 106112, sign: false });
    data.append(FP16x16 { mag: 173824, sign: false });
    data.append(FP16x16 { mag: 6104, sign: false });
    data.append(FP16x16 { mag: 92160, sign: false });
    data.append(FP16x16 { mag: 24656, sign: false });
    data.append(FP16x16 { mag: 131200, sign: false });
    data.append(FP16x16 { mag: 36064, sign: false });
    data.append(FP16x16 { mag: 80896, sign: false });
    data.append(FP16x16 { mag: 51232, sign: false });
    data.append(FP16x16 { mag: 5696, sign: false });
    data.append(FP16x16 { mag: 58304, sign: false });
    data.append(FP16x16 { mag: 64800, sign: false });
    data.append(FP16x16 { mag: 192128, sign: false });
    data.append(FP16x16 { mag: 97152, sign: false });
    data.append(FP16x16 { mag: 94784, sign: false });
    data.append(FP16x16 { mag: 173184, sign: false });
    data.append(FP16x16 { mag: 137728, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
