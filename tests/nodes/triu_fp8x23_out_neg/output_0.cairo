use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 939524096, sign: false });
    data.append(FP8x23 { mag: 478150656, sign: false });
    data.append(FP8x23 { mag: 830472192, sign: false });
    data.append(FP8x23 { mag: 629145600, sign: false });
    data.append(FP8x23 { mag: 553648128, sign: true });
    data.append(FP8x23 { mag: 377487360, sign: false });
    data.append(FP8x23 { mag: 478150656, sign: true });
    data.append(FP8x23 { mag: 150994944, sign: false });
    data.append(FP8x23 { mag: 184549376, sign: false });
    data.append(FP8x23 { mag: 411041792, sign: true });
    data.append(FP8x23 { mag: 293601280, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: false });
    data.append(FP8x23 { mag: 159383552, sign: true });
    data.append(FP8x23 { mag: 603979776, sign: false });
    data.append(FP8x23 { mag: 838860800, sign: true });
    data.append(FP8x23 { mag: 1031798784, sign: true });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 1031798784, sign: true });
    data.append(FP8x23 { mag: 1006632960, sign: false });
    data.append(FP8x23 { mag: 411041792, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
