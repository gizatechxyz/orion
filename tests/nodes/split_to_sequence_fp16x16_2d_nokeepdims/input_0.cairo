use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1376256, sign: true });
    data.append(FP16x16 { mag: 7667712, sign: true });
    data.append(FP16x16 { mag: 4849664, sign: true });
    data.append(FP16x16 { mag: 655360, sign: true });
    data.append(FP16x16 { mag: 5636096, sign: true });
    data.append(FP16x16 { mag: 7667712, sign: true });
    data.append(FP16x16 { mag: 3473408, sign: true });
    data.append(FP16x16 { mag: 2686976, sign: false });
    data.append(FP16x16 { mag: 5242880, sign: true });
    data.append(FP16x16 { mag: 6291456, sign: true });
    data.append(FP16x16 { mag: 6029312, sign: true });
    data.append(FP16x16 { mag: 2228224, sign: false });
    data.append(FP16x16 { mag: 7733248, sign: true });
    data.append(FP16x16 { mag: 3932160, sign: true });
    data.append(FP16x16 { mag: 6094848, sign: true });
    data.append(FP16x16 { mag: 2031616, sign: true });
    data.append(FP16x16 { mag: 6291456, sign: true });
    data.append(FP16x16 { mag: 6291456, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
