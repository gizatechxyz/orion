use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 146027, sign: true });
    data.append(FP16x16 { mag: 29998, sign: false });
    data.append(FP16x16 { mag: 79381, sign: true });
    data.append(FP16x16 { mag: 52502, sign: true });
    data.append(FP16x16 { mag: 141943, sign: true });
    data.append(FP16x16 { mag: 77493, sign: true });
    data.append(FP16x16 { mag: 143847, sign: true });
    data.append(FP16x16 { mag: 40577, sign: true });
    data.append(FP16x16 { mag: 15396, sign: false });
    data.append(FP16x16 { mag: 194266, sign: true });
    data.append(FP16x16 { mag: 73878, sign: true });
    data.append(FP16x16 { mag: 123262, sign: true });
    data.append(FP16x16 { mag: 106458, sign: false });
    data.append(FP16x16 { mag: 173856, sign: false });
    data.append(FP16x16 { mag: 80438, sign: true });
    data.append(FP16x16 { mag: 164829, sign: true });
    data.append(FP16x16 { mag: 89224, sign: true });
    data.append(FP16x16 { mag: 2691, sign: false });
    data.append(FP16x16 { mag: 181152, sign: true });
    data.append(FP16x16 { mag: 128977, sign: false });
    data.append(FP16x16 { mag: 78823, sign: true });
    data.append(FP16x16 { mag: 65209, sign: true });
    data.append(FP16x16 { mag: 56918, sign: false });
    data.append(FP16x16 { mag: 118799, sign: true });
    data.append(FP16x16 { mag: 179883, sign: false });
    data.append(FP16x16 { mag: 114165, sign: true });
    data.append(FP16x16 { mag: 142, sign: false });
    TensorTrait::new(shape.span(), data.span())
}