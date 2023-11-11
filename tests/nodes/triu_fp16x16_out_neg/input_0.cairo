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
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 8257536, sign: true });
    data.append(FP16x16 { mag: 1835008, sign: true });
    data.append(FP16x16 { mag: 2424832, sign: false });
    data.append(FP16x16 { mag: 1114112, sign: true });
    data.append(FP16x16 { mag: 786432, sign: true });
    data.append(FP16x16 { mag: 3604480, sign: false });
    data.append(FP16x16 { mag: 4849664, sign: true });
    data.append(FP16x16 { mag: 3735552, sign: false });
    data.append(FP16x16 { mag: 8323072, sign: true });
    data.append(FP16x16 { mag: 1703936, sign: true });
    data.append(FP16x16 { mag: 5963776, sign: false });
    data.append(FP16x16 { mag: 7208960, sign: false });
    data.append(FP16x16 { mag: 7471104, sign: false });
    data.append(FP16x16 { mag: 7929856, sign: false });
    data.append(FP16x16 { mag: 1835008, sign: true });
    data.append(FP16x16 { mag: 2228224, sign: true });
    data.append(FP16x16 { mag: 8060928, sign: true });
    data.append(FP16x16 { mag: 1638400, sign: false });
    data.append(FP16x16 { mag: 3014656, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
