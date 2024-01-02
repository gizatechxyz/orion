use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7798784, sign: true });
    data.append(FP16x16 { mag: 4587520, sign: true });
    data.append(FP16x16 { mag: 2228224, sign: true });
    data.append(FP16x16 { mag: 3407872, sign: false });
    data.append(FP16x16 { mag: 7208960, sign: true });
    data.append(FP16x16 { mag: 3735552, sign: true });
    data.append(FP16x16 { mag: 1179648, sign: true });
    data.append(FP16x16 { mag: 2752512, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: false });
    data.append(FP16x16 { mag: 5373952, sign: false });
    data.append(FP16x16 { mag: 3932160, sign: false });
    data.append(FP16x16 { mag: 3604480, sign: true });
    data.append(FP16x16 { mag: 7340032, sign: false });
    data.append(FP16x16 { mag: 5898240, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: true });
    data.append(FP16x16 { mag: 8126464, sign: true });
    data.append(FP16x16 { mag: 6029312, sign: false });
    data.append(FP16x16 { mag: 2621440, sign: false });
    data.append(FP16x16 { mag: 7667712, sign: true });
    data.append(FP16x16 { mag: 983040, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
