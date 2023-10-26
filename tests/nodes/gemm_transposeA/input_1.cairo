use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 14140, sign: false });
    data.append(FP16x16 { mag: 4427, sign: false });
    data.append(FP16x16 { mag: 61152, sign: false });
    data.append(FP16x16 { mag: 63478, sign: false });
    data.append(FP16x16 { mag: 46719, sign: false });
    data.append(FP16x16 { mag: 40516, sign: false });
    data.append(FP16x16 { mag: 5299, sign: false });
    data.append(FP16x16 { mag: 27500, sign: false });
    data.append(FP16x16 { mag: 22968, sign: false });
    data.append(FP16x16 { mag: 16628, sign: false });
    data.append(FP16x16 { mag: 14772, sign: false });
    data.append(FP16x16 { mag: 37261, sign: false });
    data.append(FP16x16 { mag: 62578, sign: false });
    data.append(FP16x16 { mag: 20150, sign: false });
    data.append(FP16x16 { mag: 38069, sign: false });
    data.append(FP16x16 { mag: 9702, sign: false });
    data.append(FP16x16 { mag: 12410, sign: false });
    data.append(FP16x16 { mag: 30336, sign: false });
    data.append(FP16x16 { mag: 65424, sign: false });
    data.append(FP16x16 { mag: 37187, sign: false });
    data.append(FP16x16 { mag: 28867, sign: false });
    data.append(FP16x16 { mag: 1671, sign: false });
    data.append(FP16x16 { mag: 57203, sign: false });
    data.append(FP16x16 { mag: 17320, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
