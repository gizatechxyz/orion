use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1056964608, sign: false });
    data.append(FP8x23 { mag: 914358272, sign: true });
    data.append(FP8x23 { mag: 645922816, sign: false });
    data.append(FP8x23 { mag: 645922816, sign: false });
    data.append(FP8x23 { mag: 310378496, sign: false });
    data.append(FP8x23 { mag: 142606336, sign: false });
    data.append(FP8x23 { mag: 411041792, sign: true });
    data.append(FP8x23 { mag: 402653184, sign: false });
    data.append(FP8x23 { mag: 184549376, sign: true });
    data.append(FP8x23 { mag: 880803840, sign: false });
    data.append(FP8x23 { mag: 25165824, sign: false });
    data.append(FP8x23 { mag: 260046848, sign: true });
    data.append(FP8x23 { mag: 402653184, sign: true });
    data.append(FP8x23 { mag: 511705088, sign: true });
    data.append(FP8x23 { mag: 100663296, sign: true });
    data.append(FP8x23 { mag: 855638016, sign: false });
    data.append(FP8x23 { mag: 478150656, sign: true });
    data.append(FP8x23 { mag: 159383552, sign: false });
    data.append(FP8x23 { mag: 201326592, sign: true });
    data.append(FP8x23 { mag: 33554432, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
