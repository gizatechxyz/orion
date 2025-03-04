use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 4284421, sign: false });
    data.append(FP8x23 { mag: 8676602, sign: true });
    data.append(FP8x23 { mag: 2893506, sign: false });
    data.append(FP8x23 { mag: 2415260, sign: true });
    data.append(FP8x23 { mag: 14482109, sign: false });
    data.append(FP8x23 { mag: 5551581, sign: true });
    data.append(FP8x23 { mag: 3917465, sign: false });
    data.append(FP8x23 { mag: 7809477, sign: false });
    data.append(FP8x23 { mag: 7627682, sign: true });
    data.append(FP8x23 { mag: 936248, sign: false });
    data.append(FP8x23 { mag: 13609040, sign: true });
    data.append(FP8x23 { mag: 2668349, sign: false });
    data.append(FP8x23 { mag: 2727640, sign: true });
    data.append(FP8x23 { mag: 27512, sign: true });
    data.append(FP8x23 { mag: 1789129, sign: true });
    data.append(FP8x23 { mag: 10845126, sign: true });
    data.append(FP8x23 { mag: 1640467, sign: false });
    data.append(FP8x23 { mag: 15969854, sign: false });
    data.append(FP8x23 { mag: 7887466, sign: true });
    data.append(FP8x23 { mag: 1601401, sign: true });
    data.append(FP8x23 { mag: 2245295, sign: false });
    data.append(FP8x23 { mag: 6773058, sign: false });
    data.append(FP8x23 { mag: 6238501, sign: false });
    data.append(FP8x23 { mag: 8850929, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
