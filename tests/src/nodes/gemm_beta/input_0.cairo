use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 14966, sign: false });
    data.append(FP16x16 { mag: 36896, sign: false });
    data.append(FP16x16 { mag: 4679, sign: false });
    data.append(FP16x16 { mag: 36625, sign: false });
    data.append(FP16x16 { mag: 48874, sign: false });
    data.append(FP16x16 { mag: 35563, sign: false });
    data.append(FP16x16 { mag: 40736, sign: false });
    data.append(FP16x16 { mag: 12321, sign: false });
    data.append(FP16x16 { mag: 42458, sign: false });
    data.append(FP16x16 { mag: 65341, sign: false });
    data.append(FP16x16 { mag: 43716, sign: false });
    data.append(FP16x16 { mag: 43328, sign: false });
    data.append(FP16x16 { mag: 7074, sign: false });
    data.append(FP16x16 { mag: 45946, sign: false });
    TensorTrait::new(shape.span(), data.span())
}