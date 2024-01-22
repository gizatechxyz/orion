use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 679477248, sign: true });
    data.append(FP8x23 { mag: 864026624, sign: true });
    data.append(FP8x23 { mag: 612368384, sign: false });
    data.append(FP8x23 { mag: 226492416, sign: false });
    data.append(FP8x23 { mag: 520093696, sign: false });
    data.append(FP8x23 { mag: 75497472, sign: true });
    data.append(FP8x23 { mag: 939524096, sign: true });
    data.append(FP8x23 { mag: 75497472, sign: true });
    data.append(FP8x23 { mag: 872415232, sign: false });
    data.append(FP8x23 { mag: 637534208, sign: true });
    data.append(FP8x23 { mag: 142606336, sign: false });
    data.append(FP8x23 { mag: 92274688, sign: false });
    data.append(FP8x23 { mag: 947912704, sign: true });
    data.append(FP8x23 { mag: 385875968, sign: false });
    data.append(FP8x23 { mag: 494927872, sign: false });
    data.append(FP8x23 { mag: 939524096, sign: false });
    data.append(FP8x23 { mag: 452984832, sign: true });
    data.append(FP8x23 { mag: 562036736, sign: true });
    data.append(FP8x23 { mag: 587202560, sign: false });
    data.append(FP8x23 { mag: 838860800, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
