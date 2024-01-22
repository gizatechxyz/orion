use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 973078528, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 452984832, sign: false });
    data.append(FP8x23 { mag: 83886080, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 369098752, sign: true });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 738197504, sign: false });
    data.append(FP8x23 { mag: 478150656, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 528482304, sign: false });
    data.append(FP8x23 { mag: 629145600, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 67108864, sign: false });
    data.append(FP8x23 { mag: 184549376, sign: false });
    data.append(FP8x23 { mag: 243269632, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
