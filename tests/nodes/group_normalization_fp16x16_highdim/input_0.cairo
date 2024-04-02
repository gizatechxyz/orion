use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 24211, sign: false });
    data.append(FP16x16 { mag: 1461, sign: true });
    data.append(FP16x16 { mag: 30715, sign: true });
    data.append(FP16x16 { mag: 43111, sign: false });
    data.append(FP16x16 { mag: 84100, sign: true });
    data.append(FP16x16 { mag: 18726, sign: false });
    data.append(FP16x16 { mag: 21507, sign: false });
    data.append(FP16x16 { mag: 6693, sign: true });
    data.append(FP16x16 { mag: 21854, sign: true });
    data.append(FP16x16 { mag: 27738, sign: false });
    data.append(FP16x16 { mag: 37980, sign: false });
    data.append(FP16x16 { mag: 5193, sign: false });
    data.append(FP16x16 { mag: 114188, sign: true });
    data.append(FP16x16 { mag: 35804, sign: true });
    data.append(FP16x16 { mag: 72580, sign: true });
    data.append(FP16x16 { mag: 1419, sign: false });
    data.append(FP16x16 { mag: 23322, sign: true });
    data.append(FP16x16 { mag: 36613, sign: true });
    data.append(FP16x16 { mag: 55359, sign: true });
    data.append(FP16x16 { mag: 48867, sign: true });
    data.append(FP16x16 { mag: 97366, sign: false });
    data.append(FP16x16 { mag: 137268, sign: false });
    data.append(FP16x16 { mag: 35341, sign: false });
    data.append(FP16x16 { mag: 59210, sign: true });
    data.append(FP16x16 { mag: 48675, sign: true });
    data.append(FP16x16 { mag: 26116, sign: false });
    data.append(FP16x16 { mag: 71355, sign: false });
    data.append(FP16x16 { mag: 27365, sign: false });
    data.append(FP16x16 { mag: 19618, sign: false });
    data.append(FP16x16 { mag: 80595, sign: true });
    data.append(FP16x16 { mag: 54553, sign: false });
    data.append(FP16x16 { mag: 25798, sign: false });
    data.append(FP16x16 { mag: 118183, sign: true });
    data.append(FP16x16 { mag: 140009, sign: true });
    data.append(FP16x16 { mag: 6148, sign: false });
    data.append(FP16x16 { mag: 19513, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
