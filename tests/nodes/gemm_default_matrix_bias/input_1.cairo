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
    data.append(FP16x16 { mag: 47776, sign: false });
    data.append(FP16x16 { mag: 8702, sign: false });
    data.append(FP16x16 { mag: 39764, sign: false });
    data.append(FP16x16 { mag: 21672, sign: false });
    data.append(FP16x16 { mag: 34121, sign: false });
    data.append(FP16x16 { mag: 60787, sign: false });
    data.append(FP16x16 { mag: 50462, sign: false });
    data.append(FP16x16 { mag: 61510, sign: false });
    data.append(FP16x16 { mag: 39048, sign: false });
    data.append(FP16x16 { mag: 32834, sign: false });
    data.append(FP16x16 { mag: 57152, sign: false });
    data.append(FP16x16 { mag: 4001, sign: false });
    data.append(FP16x16 { mag: 37122, sign: false });
    data.append(FP16x16 { mag: 45910, sign: false });
    data.append(FP16x16 { mag: 22021, sign: false });
    data.append(FP16x16 { mag: 10298, sign: false });
    data.append(FP16x16 { mag: 33089, sign: false });
    data.append(FP16x16 { mag: 35378, sign: false });
    data.append(FP16x16 { mag: 1834, sign: false });
    data.append(FP16x16 { mag: 22627, sign: false });
    data.append(FP16x16 { mag: 37576, sign: false });
    data.append(FP16x16 { mag: 57351, sign: false });
    data.append(FP16x16 { mag: 22814, sign: false });
    data.append(FP16x16 { mag: 60423, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
