use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 27243, sign: false });
    data.append(FP16x16 { mag: 36813, sign: false });
    data.append(FP16x16 { mag: 37987, sign: false });
    data.append(FP16x16 { mag: 26203, sign: false });
    data.append(FP16x16 { mag: 15897, sign: false });
    data.append(FP16x16 { mag: 22608, sign: false });
    data.append(FP16x16 { mag: 34035, sign: false });
    data.append(FP16x16 { mag: 35198, sign: false });
    data.append(FP16x16 { mag: 24134, sign: false });
    data.append(FP16x16 { mag: 15535, sign: false });
    data.append(FP16x16 { mag: 20351, sign: false });
    data.append(FP16x16 { mag: 30911, sign: false });
    data.append(FP16x16 { mag: 39882, sign: false });
    data.append(FP16x16 { mag: 23721, sign: false });
    data.append(FP16x16 { mag: 15499, sign: false });
    TensorTrait::new(shape.span(), data.span())
}