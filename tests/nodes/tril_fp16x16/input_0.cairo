use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7208960, sign: true });
    data.append(FP16x16 { mag: 1507328, sign: false });
    data.append(FP16x16 { mag: 7077888, sign: true });
    data.append(FP16x16 { mag: 2424832, sign: true });
    data.append(FP16x16 { mag: 3932160, sign: false });
    data.append(FP16x16 { mag: 2686976, sign: false });
    data.append(FP16x16 { mag: 1638400, sign: true });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 2818048, sign: false });
    data.append(FP16x16 { mag: 1900544, sign: true });
    data.append(FP16x16 { mag: 5701632, sign: true });
    data.append(FP16x16 { mag: 2490368, sign: false });
    data.append(FP16x16 { mag: 6881280, sign: false });
    data.append(FP16x16 { mag: 1900544, sign: false });
    data.append(FP16x16 { mag: 4849664, sign: false });
    data.append(FP16x16 { mag: 6160384, sign: false });
    data.append(FP16x16 { mag: 4784128, sign: false });
    data.append(FP16x16 { mag: 393216, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 5570560, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
