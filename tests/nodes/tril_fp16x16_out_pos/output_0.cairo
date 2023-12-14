use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 720896, sign: false });
    data.append(FP16x16 { mag: 5046272, sign: false });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 5832704, sign: false });
    data.append(FP16x16 { mag: 7340032, sign: true });
    data.append(FP16x16 { mag: 8192000, sign: false });
    data.append(FP16x16 { mag: 6094848, sign: false });
    data.append(FP16x16 { mag: 6488064, sign: true });
    data.append(FP16x16 { mag: 458752, sign: false });
    data.append(FP16x16 { mag: 7405568, sign: true });
    data.append(FP16x16 { mag: 5373952, sign: false });
    data.append(FP16x16 { mag: 4521984, sign: true });
    data.append(FP16x16 { mag: 7667712, sign: true });
    data.append(FP16x16 { mag: 3014656, sign: false });
    data.append(FP16x16 { mag: 1900544, sign: false });
    data.append(FP16x16 { mag: 3211264, sign: false });
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 6815744, sign: false });
    data.append(FP16x16 { mag: 5963776, sign: true });
    data.append(FP16x16 { mag: 3866624, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
