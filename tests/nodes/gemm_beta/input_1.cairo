use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(7);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 51391, sign: false });
    data.append(FP16x16 { mag: 22014, sign: false });
    data.append(FP16x16 { mag: 33442, sign: false });
    data.append(FP16x16 { mag: 24116, sign: false });
    data.append(FP16x16 { mag: 49410, sign: false });
    data.append(FP16x16 { mag: 60215, sign: false });
    data.append(FP16x16 { mag: 9310, sign: false });
    data.append(FP16x16 { mag: 20950, sign: false });
    data.append(FP16x16 { mag: 20541, sign: false });
    data.append(FP16x16 { mag: 21583, sign: false });
    data.append(FP16x16 { mag: 28565, sign: false });
    data.append(FP16x16 { mag: 41677, sign: false });
    data.append(FP16x16 { mag: 18308, sign: false });
    data.append(FP16x16 { mag: 25095, sign: false });
    data.append(FP16x16 { mag: 44238, sign: false });
    data.append(FP16x16 { mag: 27465, sign: false });
    data.append(FP16x16 { mag: 30581, sign: false });
    data.append(FP16x16 { mag: 41045, sign: false });
    data.append(FP16x16 { mag: 46018, sign: false });
    data.append(FP16x16 { mag: 17358, sign: false });
    data.append(FP16x16 { mag: 50102, sign: false });
    data.append(FP16x16 { mag: 16577, sign: false });
    data.append(FP16x16 { mag: 16374, sign: false });
    data.append(FP16x16 { mag: 54251, sign: false });
    data.append(FP16x16 { mag: 46337, sign: false });
    data.append(FP16x16 { mag: 15187, sign: false });
    data.append(FP16x16 { mag: 25652, sign: false });
    data.append(FP16x16 { mag: 20892, sign: false });
    TensorTrait::new(shape.span(), data.span())
}