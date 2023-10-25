use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 60981, sign: false });
    data.append(FP16x16 { mag: 58843, sign: false });
    data.append(FP16x16 { mag: 19404, sign: false });
    data.append(FP16x16 { mag: 56768, sign: false });
    data.append(FP16x16 { mag: 2442, sign: false });
    data.append(FP16x16 { mag: 45529, sign: false });
    data.append(FP16x16 { mag: 1800, sign: false });
    data.append(FP16x16 { mag: 38751, sign: false });
    data.append(FP16x16 { mag: 29332, sign: false });
    data.append(FP16x16 { mag: 17874, sign: false });
    data.append(FP16x16 { mag: 39405, sign: false });
    data.append(FP16x16 { mag: 7286, sign: false });
    data.append(FP16x16 { mag: 23687, sign: false });
    data.append(FP16x16 { mag: 7092, sign: false });
    data.append(FP16x16 { mag: 20015, sign: false });
    data.append(FP16x16 { mag: 26356, sign: false });
    data.append(FP16x16 { mag: 49636, sign: false });
    data.append(FP16x16 { mag: 54933, sign: false });
    TensorTrait::new(shape.span(), data.span())
}