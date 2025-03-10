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
    data.append(FP8x23 { mag: 39029932, sign: true });
    data.append(FP8x23 { mag: 34440380, sign: true });
    data.append(FP8x23 { mag: 48251608, sign: true });
    data.append(FP8x23 { mag: 31293496, sign: true });
    data.append(FP8x23 { mag: 33462898, sign: true });
    data.append(FP8x23 { mag: 50195752, sign: true });
    data.append(FP8x23 { mag: 34697416, sign: true });
    data.append(FP8x23 { mag: 37249188, sign: true });
    data.append(FP8x23 { mag: 40731904, sign: true });
    data.append(FP8x23 { mag: 34401020, sign: true });
    data.append(FP8x23 { mag: 44109680, sign: true });
    data.append(FP8x23 { mag: 25328908, sign: true });
    data.append(FP8x23 { mag: 37090832, sign: true });
    data.append(FP8x23 { mag: 49931188, sign: true });
    data.append(FP8x23 { mag: 31626200, sign: true });
    data.append(FP8x23 { mag: 44868080, sign: true });
    data.append(FP8x23 { mag: 38649832, sign: true });
    data.append(FP8x23 { mag: 42030664, sign: true });
    data.append(FP8x23 { mag: 44171392, sign: true });
    data.append(FP8x23 { mag: 26818130, sign: true });
    data.append(FP8x23 { mag: 28059510, sign: true });
    data.append(FP8x23 { mag: 27430244, sign: true });
    data.append(FP8x23 { mag: 33228426, sign: true });
    data.append(FP8x23 { mag: 49704420, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
