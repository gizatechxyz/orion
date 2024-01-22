use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 394264576, sign: true });
    data.append(FP8x23 { mag: 75497472, sign: true });
    data.append(FP8x23 { mag: 763363328, sign: true });
    data.append(FP8x23 { mag: 998244352, sign: true });
    data.append(FP8x23 { mag: 645922816, sign: true });
    data.append(FP8x23 { mag: 75497472, sign: true });
    data.append(FP8x23 { mag: 889192448, sign: true });
    data.append(FP8x23 { mag: 301989888, sign: false });
    data.append(FP8x23 { mag: 67108864, sign: false });
    data.append(FP8x23 { mag: 1006632960, sign: true });
    data.append(FP8x23 { mag: 134217728, sign: true });
    data.append(FP8x23 { mag: 780140544, sign: true });
    data.append(FP8x23 { mag: 687865856, sign: false });
    data.append(FP8x23 { mag: 998244352, sign: false });
    data.append(FP8x23 { mag: 511705088, sign: false });
    data.append(FP8x23 { mag: 847249408, sign: false });
    data.append(FP8x23 { mag: 343932928, sign: false });
    data.append(FP8x23 { mag: 553648128, sign: true });
    data.append(FP8x23 { mag: 956301312, sign: true });
    data.append(FP8x23 { mag: 998244352, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
