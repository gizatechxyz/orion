use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 729808896, sign: true });
    data.append(FP8x23 { mag: 587202560, sign: false });
    data.append(FP8x23 { mag: 998244352, sign: true });
    data.append(FP8x23 { mag: 620756992, sign: true });
    data.append(FP8x23 { mag: 897581056, sign: true });
    data.append(FP8x23 { mag: 536870912, sign: false });
    data.append(FP8x23 { mag: 805306368, sign: false });
    data.append(FP8x23 { mag: 922746880, sign: true });
    data.append(FP8x23 { mag: 109051904, sign: true });
    data.append(FP8x23 { mag: 452984832, sign: true });
    data.append(FP8x23 { mag: 293601280, sign: false });
    data.append(FP8x23 { mag: 8388608, sign: false });
    data.append(FP8x23 { mag: 109051904, sign: true });
    data.append(FP8x23 { mag: 897581056, sign: false });
    data.append(FP8x23 { mag: 511705088, sign: false });
    data.append(FP8x23 { mag: 218103808, sign: true });
    data.append(FP8x23 { mag: 847249408, sign: true });
    data.append(FP8x23 { mag: 268435456, sign: true });
    data.append(FP8x23 { mag: 369098752, sign: true });
    data.append(FP8x23 { mag: 117440512, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
