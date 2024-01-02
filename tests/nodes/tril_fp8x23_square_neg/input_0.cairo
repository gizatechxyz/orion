use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 922746880, sign: true });
    data.append(FP8x23 { mag: 494927872, sign: false });
    data.append(FP8x23 { mag: 553648128, sign: true });
    data.append(FP8x23 { mag: 998244352, sign: false });
    data.append(FP8x23 { mag: 469762048, sign: false });
    data.append(FP8x23 { mag: 687865856, sign: false });
    data.append(FP8x23 { mag: 469762048, sign: false });
    data.append(FP8x23 { mag: 33554432, sign: false });
    data.append(FP8x23 { mag: 201326592, sign: false });
    data.append(FP8x23 { mag: 511705088, sign: false });
    data.append(FP8x23 { mag: 545259520, sign: false });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 360710144, sign: false });
    data.append(FP8x23 { mag: 889192448, sign: false });
    data.append(FP8x23 { mag: 461373440, sign: true });
    data.append(FP8x23 { mag: 771751936, sign: false });
    data.append(FP8x23 { mag: 830472192, sign: false });
    data.append(FP8x23 { mag: 520093696, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
