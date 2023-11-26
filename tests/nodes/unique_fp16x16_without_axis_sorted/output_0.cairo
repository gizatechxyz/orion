use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(27);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 167936, sign: false });
    data.append(FP16x16 { mag: 103168, sign: false });
    data.append(FP16x16 { mag: 73664, sign: false });
    data.append(FP16x16 { mag: 187008, sign: false });
    data.append(FP16x16 { mag: 81280, sign: false });
    data.append(FP16x16 { mag: 7340, sign: false });
    data.append(FP16x16 { mag: 83776, sign: false });
    data.append(FP16x16 { mag: 97088, sign: false });
    data.append(FP16x16 { mag: 96128, sign: false });
    data.append(FP16x16 { mag: 81024, sign: false });
    data.append(FP16x16 { mag: 10896, sign: false });
    data.append(FP16x16 { mag: 56128, sign: false });
    data.append(FP16x16 { mag: 85440, sign: false });
    data.append(FP16x16 { mag: 82624, sign: false });
    data.append(FP16x16 { mag: 93504, sign: false });
    data.append(FP16x16 { mag: 146304, sign: false });
    data.append(FP16x16 { mag: 23376, sign: false });
    data.append(FP16x16 { mag: 175744, sign: false });
    data.append(FP16x16 { mag: 122944, sign: false });
    data.append(FP16x16 { mag: 58176, sign: false });
    data.append(FP16x16 { mag: 65504, sign: false });
    data.append(FP16x16 { mag: 167168, sign: false });
    data.append(FP16x16 { mag: 189440, sign: false });
    data.append(FP16x16 { mag: 52800, sign: false });
    data.append(FP16x16 { mag: 14408, sign: false });
    data.append(FP16x16 { mag: 70272, sign: false });
    data.append(FP16x16 { mag: 62752, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
