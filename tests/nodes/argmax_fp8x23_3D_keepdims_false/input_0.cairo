use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 444596224, sign: false });
    data.append(FP8x23 { mag: 369098752, sign: false });
    data.append(FP8x23 { mag: 1056964608, sign: false });
    data.append(FP8x23 { mag: 234881024, sign: false });
    data.append(FP8x23 { mag: 159383552, sign: true });
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 897581056, sign: true });
    data.append(FP8x23 { mag: 327155712, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
