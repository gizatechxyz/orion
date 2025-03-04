use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 11115845, sign: false });
    data.append(FP8x23 { mag: 3642026, sign: true });
    data.append(FP8x23 { mag: 5289849, sign: false });
    data.append(FP8x23 { mag: 1125422, sign: false });
    data.append(FP8x23 { mag: 11666879, sign: true });
    data.append(FP8x23 { mag: 13888535, sign: false });
    data.append(FP8x23 { mag: 541416, sign: true });
    data.append(FP8x23 { mag: 4786489, sign: true });
    data.append(FP8x23 { mag: 2043357, sign: false });
    data.append(FP8x23 { mag: 6628811, sign: true });
    data.append(FP8x23 { mag: 13597537, sign: true });
    data.append(FP8x23 { mag: 10641523, sign: false });
    data.append(FP8x23 { mag: 8042851, sign: true });
    data.append(FP8x23 { mag: 17146492, sign: false });
    data.append(FP8x23 { mag: 7248135, sign: true });
    data.append(FP8x23 { mag: 11945658, sign: false });
    data.append(FP8x23 { mag: 18376966, sign: false });
    data.append(FP8x23 { mag: 8743412, sign: true });
    data.append(FP8x23 { mag: 4070581, sign: false });
    data.append(FP8x23 { mag: 7482028, sign: true });
    data.append(FP8x23 { mag: 274148, sign: true });
    data.append(FP8x23 { mag: 2408931, sign: true });
    data.append(FP8x23 { mag: 7133879, sign: true });
    data.append(FP8x23 { mag: 16624649, sign: false });
    data.append(FP8x23 { mag: 1251276, sign: true });
    data.append(FP8x23 { mag: 7034681, sign: true });
    data.append(FP8x23 { mag: 4453781, sign: false });
    data.append(FP8x23 { mag: 5772860, sign: false });
    data.append(FP8x23 { mag: 4097435, sign: true });
    data.append(FP8x23 { mag: 12496413, sign: false });
    data.append(FP8x23 { mag: 7414602, sign: true });
    data.append(FP8x23 { mag: 2179771, sign: false });
    data.append(FP8x23 { mag: 3618895, sign: true });
    data.append(FP8x23 { mag: 6155062, sign: false });
    data.append(FP8x23 { mag: 7292676, sign: false });
    data.append(FP8x23 { mag: 2080689, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
