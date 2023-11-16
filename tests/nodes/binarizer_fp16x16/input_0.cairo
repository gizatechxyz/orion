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
    data.append(FP16x16 { mag: 116651, sign: true });
    data.append(FP16x16 { mag: 45255, sign: false });
    data.append(FP16x16 { mag: 132067, sign: true });
    data.append(FP16x16 { mag: 83631, sign: true });
    data.append(FP16x16 { mag: 151971, sign: false });
    data.append(FP16x16 { mag: 46816, sign: true });
    data.append(FP16x16 { mag: 4353, sign: false });
    data.append(FP16x16 { mag: 188531, sign: false });
    data.append(FP16x16 { mag: 104593, sign: false });
    data.append(FP16x16 { mag: 41997, sign: true });
    data.append(FP16x16 { mag: 127090, sign: false });
    data.append(FP16x16 { mag: 79053, sign: false });
    data.append(FP16x16 { mag: 139184, sign: true });
    data.append(FP16x16 { mag: 439, sign: false });
    data.append(FP16x16 { mag: 115688, sign: false });
    data.append(FP16x16 { mag: 184428, sign: true });
    data.append(FP16x16 { mag: 161855, sign: true });
    data.append(FP16x16 { mag: 42551, sign: true });
    data.append(FP16x16 { mag: 40207, sign: false });
    data.append(FP16x16 { mag: 72062, sign: false });
    data.append(FP16x16 { mag: 195673, sign: false });
    data.append(FP16x16 { mag: 27413, sign: true });
    data.append(FP16x16 { mag: 173891, sign: true });
    data.append(FP16x16 { mag: 69697, sign: true });
    data.append(FP16x16 { mag: 162632, sign: false });
    data.append(FP16x16 { mag: 53852, sign: true });
    data.append(FP16x16 { mag: 93775, sign: false });
    TensorTrait::new(shape.span(), data.span())
}