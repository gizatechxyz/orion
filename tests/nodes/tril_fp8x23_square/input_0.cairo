use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 822083584, sign: true });
    data.append(FP8x23 { mag: 125829120, sign: false });
    data.append(FP8x23 { mag: 184549376, sign: true });
    data.append(FP8x23 { mag: 335544320, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: false });
    data.append(FP8x23 { mag: 369098752, sign: false });
    data.append(FP8x23 { mag: 687865856, sign: true });
    data.append(FP8x23 { mag: 192937984, sign: false });
    data.append(FP8x23 { mag: 192937984, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 1023410176, sign: false });
    data.append(FP8x23 { mag: 679477248, sign: true });
    data.append(FP8x23 { mag: 1015021568, sign: false });
    data.append(FP8x23 { mag: 637534208, sign: true });
    data.append(FP8x23 { mag: 109051904, sign: true });
    data.append(FP8x23 { mag: 335544320, sign: true });
    data.append(FP8x23 { mag: 243269632, sign: false });
    data.append(FP8x23 { mag: 511705088, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
