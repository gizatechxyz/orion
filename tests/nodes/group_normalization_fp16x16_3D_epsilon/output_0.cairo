use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1117, sign: true });
    data.append(FP16x16 { mag: 22727, sign: false });
    data.append(FP16x16 { mag: 75951, sign: true });
    data.append(FP16x16 { mag: 54110, sign: true });
    data.append(FP16x16 { mag: 54043, sign: false });
    data.append(FP16x16 { mag: 16542, sign: false });
    data.append(FP16x16 { mag: 16160, sign: true });
    data.append(FP16x16 { mag: 128441, sign: false });
    data.append(FP16x16 { mag: 2535, sign: false });
    data.append(FP16x16 { mag: 50326, sign: true });
    data.append(FP16x16 { mag: 66162, sign: true });
    data.append(FP16x16 { mag: 51575, sign: true });
    data.append(FP16x16 { mag: 86752, sign: false });
    data.append(FP16x16 { mag: 76318, sign: false });
    data.append(FP16x16 { mag: 154125, sign: false });
    data.append(FP16x16 { mag: 118295, sign: false });
    data.append(FP16x16 { mag: 115847, sign: false });
    data.append(FP16x16 { mag: 15491, sign: true });
    data.append(FP16x16 { mag: 76374, sign: true });
    data.append(FP16x16 { mag: 67672, sign: true });
    data.append(FP16x16 { mag: 18204, sign: false });
    data.append(FP16x16 { mag: 12937, sign: false });
    data.append(FP16x16 { mag: 54432, sign: false });
    data.append(FP16x16 { mag: 10448, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
