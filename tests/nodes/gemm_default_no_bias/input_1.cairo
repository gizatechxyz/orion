use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 50500, sign: false });
    data.append(FP16x16 { mag: 17886, sign: false });
    data.append(FP16x16 { mag: 46985, sign: false });
    data.append(FP16x16 { mag: 55588, sign: false });
    data.append(FP16x16 { mag: 13076, sign: false });
    data.append(FP16x16 { mag: 60436, sign: false });
    data.append(FP16x16 { mag: 39821, sign: false });
    data.append(FP16x16 { mag: 26415, sign: false });
    data.append(FP16x16 { mag: 21305, sign: false });
    data.append(FP16x16 { mag: 14320, sign: false });
    data.append(FP16x16 { mag: 28448, sign: false });
    data.append(FP16x16 { mag: 25828, sign: false });
    data.append(FP16x16 { mag: 47472, sign: false });
    data.append(FP16x16 { mag: 52266, sign: false });
    data.append(FP16x16 { mag: 7390, sign: false });
    data.append(FP16x16 { mag: 56380, sign: false });
    data.append(FP16x16 { mag: 13296, sign: false });
    data.append(FP16x16 { mag: 59748, sign: false });
    data.append(FP16x16 { mag: 8798, sign: false });
    data.append(FP16x16 { mag: 32105, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
