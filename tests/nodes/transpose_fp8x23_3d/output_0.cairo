use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 645922816, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 553648128, sign: false });
    data.append(FP8x23 { mag: 335544320, sign: false });
    data.append(FP8x23 { mag: 629145600, sign: false });
    data.append(FP8x23 { mag: 1006632960, sign: false });
    data.append(FP8x23 { mag: 301989888, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
