use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 783, sign: true });
    data.append(FP16x16 { mag: 136168, sign: true });
    data.append(FP16x16 { mag: 71974, sign: false });
    data.append(FP16x16 { mag: 179326, sign: false });
    data.append(FP16x16 { mag: 152653, sign: true });
    data.append(FP16x16 { mag: 26718, sign: false });
    data.append(FP16x16 { mag: 37215, sign: false });
    data.append(FP16x16 { mag: 183129, sign: true });
    data.append(FP16x16 { mag: 118922, sign: true });
    data.append(FP16x16 { mag: 32220, sign: false });
    data.append(FP16x16 { mag: 107283, sign: false });
    data.append(FP16x16 { mag: 63927, sign: false });
    data.append(FP16x16 { mag: 169946, sign: true });
    data.append(FP16x16 { mag: 159111, sign: true });
    data.append(FP16x16 { mag: 187330, sign: false });
    data.append(FP16x16 { mag: 112385, sign: true });
    data.append(FP16x16 { mag: 145842, sign: false });
    data.append(FP16x16 { mag: 34532, sign: false });
    data.append(FP16x16 { mag: 117182, sign: true });
    data.append(FP16x16 { mag: 23252, sign: false });
    data.append(FP16x16 { mag: 78368, sign: true });
    data.append(FP16x16 { mag: 137560, sign: false });
    data.append(FP16x16 { mag: 157981, sign: false });
    data.append(FP16x16 { mag: 89138, sign: true });
    data.append(FP16x16 { mag: 108598, sign: true });
    data.append(FP16x16 { mag: 144474, sign: true });
    data.append(FP16x16 { mag: 118439, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
