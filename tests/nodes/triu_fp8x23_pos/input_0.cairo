use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 33554432, sign: false });
    data.append(FP8x23 { mag: 721420288, sign: true });
    data.append(FP8x23 { mag: 125829120, sign: true });
    data.append(FP8x23 { mag: 1040187392, sign: false });
    data.append(FP8x23 { mag: 1006632960, sign: true });
    data.append(FP8x23 { mag: 92274688, sign: false });
    data.append(FP8x23 { mag: 788529152, sign: true });
    data.append(FP8x23 { mag: 478150656, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 125829120, sign: false });
    data.append(FP8x23 { mag: 452984832, sign: false });
    data.append(FP8x23 { mag: 503316480, sign: true });
    data.append(FP8x23 { mag: 629145600, sign: true });
    data.append(FP8x23 { mag: 713031680, sign: true });
    data.append(FP8x23 { mag: 243269632, sign: true });
    data.append(FP8x23 { mag: 58720256, sign: true });
    data.append(FP8x23 { mag: 536870912, sign: true });
    data.append(FP8x23 { mag: 167772160, sign: true });
    data.append(FP8x23 { mag: 452984832, sign: true });
    data.append(FP8x23 { mag: 260046848, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
