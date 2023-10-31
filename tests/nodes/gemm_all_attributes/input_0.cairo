use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 25442, sign: false });
    data.append(FP16x16 { mag: 21621, sign: false });
    data.append(FP16x16 { mag: 20558, sign: false });
    data.append(FP16x16 { mag: 63086, sign: false });
    data.append(FP16x16 { mag: 42888, sign: false });
    data.append(FP16x16 { mag: 5836, sign: false });
    data.append(FP16x16 { mag: 36243, sign: false });
    data.append(FP16x16 { mag: 31967, sign: false });
    data.append(FP16x16 { mag: 64085, sign: false });
    data.append(FP16x16 { mag: 26601, sign: false });
    data.append(FP16x16 { mag: 40779, sign: false });
    data.append(FP16x16 { mag: 41935, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
