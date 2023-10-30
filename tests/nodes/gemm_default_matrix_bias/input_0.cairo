use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 54171, sign: false });
    data.append(FP16x16 { mag: 576, sign: false });
    data.append(FP16x16 { mag: 51387, sign: false });
    data.append(FP16x16 { mag: 37774, sign: false });
    data.append(FP16x16 { mag: 47415, sign: false });
    data.append(FP16x16 { mag: 30278, sign: false });
    data.append(FP16x16 { mag: 35329, sign: false });
    data.append(FP16x16 { mag: 56770, sign: false });
    data.append(FP16x16 { mag: 29001, sign: false });
    data.append(FP16x16 { mag: 19387, sign: false });
    data.append(FP16x16 { mag: 16747, sign: false });
    data.append(FP16x16 { mag: 42410, sign: false });
    data.append(FP16x16 { mag: 53192, sign: false });
    data.append(FP16x16 { mag: 30490, sign: false });
    data.append(FP16x16 { mag: 55512, sign: false });
    data.append(FP16x16 { mag: 63983, sign: false });
    data.append(FP16x16 { mag: 45579, sign: false });
    data.append(FP16x16 { mag: 12475, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
