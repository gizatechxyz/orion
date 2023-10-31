use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 28356, sign: false });
    data.append(FP16x16 { mag: 19354, sign: false });
    data.append(FP16x16 { mag: 2749, sign: false });
    data.append(FP16x16 { mag: 24026, sign: false });
    data.append(FP16x16 { mag: 11157, sign: false });
    data.append(FP16x16 { mag: 62112, sign: false });
    data.append(FP16x16 { mag: 8802, sign: false });
    data.append(FP16x16 { mag: 40701, sign: false });
    data.append(FP16x16 { mag: 11492, sign: false });
    data.append(FP16x16 { mag: 56717, sign: false });
    data.append(FP16x16 { mag: 12174, sign: false });
    data.append(FP16x16 { mag: 37607, sign: false });
    data.append(FP16x16 { mag: 18568, sign: false });
    data.append(FP16x16 { mag: 56759, sign: false });
    data.append(FP16x16 { mag: 17097, sign: false });
    data.append(FP16x16 { mag: 39335, sign: false });
    data.append(FP16x16 { mag: 50570, sign: false });
    data.append(FP16x16 { mag: 54411, sign: false });
    data.append(FP16x16 { mag: 25640, sign: false });
    data.append(FP16x16 { mag: 55921, sign: false });
    data.append(FP16x16 { mag: 37203, sign: false });
    data.append(FP16x16 { mag: 10548, sign: false });
    data.append(FP16x16 { mag: 48030, sign: false });
    data.append(FP16x16 { mag: 37338, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
