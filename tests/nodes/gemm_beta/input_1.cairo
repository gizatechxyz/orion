use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(7);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 61572, sign: false });
    data.append(FP16x16 { mag: 45985, sign: false });
    data.append(FP16x16 { mag: 19404, sign: false });
    data.append(FP16x16 { mag: 4011, sign: false });
    data.append(FP16x16 { mag: 4824, sign: false });
    data.append(FP16x16 { mag: 17368, sign: false });
    data.append(FP16x16 { mag: 26865, sign: false });
    data.append(FP16x16 { mag: 19036, sign: false });
    data.append(FP16x16 { mag: 29179, sign: false });
    data.append(FP16x16 { mag: 27268, sign: false });
    data.append(FP16x16 { mag: 11806, sign: false });
    data.append(FP16x16 { mag: 41521, sign: false });
    data.append(FP16x16 { mag: 51714, sign: false });
    data.append(FP16x16 { mag: 9380, sign: false });
    data.append(FP16x16 { mag: 637, sign: false });
    data.append(FP16x16 { mag: 34737, sign: false });
    data.append(FP16x16 { mag: 6709, sign: false });
    data.append(FP16x16 { mag: 51346, sign: false });
    data.append(FP16x16 { mag: 24730, sign: false });
    data.append(FP16x16 { mag: 2358, sign: false });
    data.append(FP16x16 { mag: 12807, sign: false });
    data.append(FP16x16 { mag: 41228, sign: false });
    data.append(FP16x16 { mag: 60913, sign: false });
    data.append(FP16x16 { mag: 15947, sign: false });
    data.append(FP16x16 { mag: 45496, sign: false });
    data.append(FP16x16 { mag: 60217, sign: false });
    data.append(FP16x16 { mag: 51117, sign: false });
    data.append(FP16x16 { mag: 52177, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
