use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2752512, sign: true });
    data.append(FP16x16 { mag: 5898240, sign: true });
    data.append(FP16x16 { mag: 7667712, sign: true });
    data.append(FP16x16 { mag: 7798784, sign: true });
    data.append(FP16x16 { mag: 1376256, sign: true });
    data.append(FP16x16 { mag: 5242880, sign: false });
    data.append(FP16x16 { mag: 2228224, sign: true });
    data.append(FP16x16 { mag: 4849664, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 1376256, sign: false });
    data.append(FP16x16 { mag: 6291456, sign: true });
    data.append(FP16x16 { mag: 7471104, sign: true });
    data.append(FP16x16 { mag: 4390912, sign: false });
    data.append(FP16x16 { mag: 8060928, sign: true });
    data.append(FP16x16 { mag: 6619136, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 720896, sign: true });
    data.append(FP16x16 { mag: 5373952, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: false });
    data.append(FP16x16 { mag: 3997696, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
