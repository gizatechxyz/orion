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
    data.append(FP16x16 { mag: 157440, sign: true });
    data.append(FP16x16 { mag: 1772, sign: true });
    data.append(FP16x16 { mag: 104448, sign: false });
    data.append(FP16x16 { mag: 2094, sign: true });
    data.append(FP16x16 { mag: 63936, sign: false });
    data.append(FP16x16 { mag: 120448, sign: false });
    data.append(FP16x16 { mag: 72704, sign: false });
    data.append(FP16x16 { mag: 15552, sign: true });
    data.append(FP16x16 { mag: 44992, sign: false });
    data.append(FP16x16 { mag: 94208, sign: false });
    data.append(FP16x16 { mag: 93312, sign: true });
    data.append(FP16x16 { mag: 29024, sign: true });
    data.append(FP16x16 { mag: 52448, sign: true });
    data.append(FP16x16 { mag: 176000, sign: true });
    data.append(FP16x16 { mag: 98624, sign: true });
    data.append(FP16x16 { mag: 164480, sign: true });
    data.append(FP16x16 { mag: 140544, sign: true });
    data.append(FP16x16 { mag: 125760, sign: true });
    data.append(FP16x16 { mag: 64800, sign: false });
    data.append(FP16x16 { mag: 89664, sign: true });
    data.append(FP16x16 { mag: 125376, sign: false });
    data.append(FP16x16 { mag: 89408, sign: true });
    data.append(FP16x16 { mag: 93376, sign: false });
    data.append(FP16x16 { mag: 26320, sign: true });
    data.append(FP16x16 { mag: 2108, sign: true });
    data.append(FP16x16 { mag: 131968, sign: false });
    data.append(FP16x16 { mag: 69568, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
