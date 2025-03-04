use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(1);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 15064009, sign: true });
    data.append(FP8x23 { mag: 6585725, sign: true });
    data.append(FP8x23 { mag: 24856658, sign: true });
    data.append(FP8x23 { mag: 14681511, sign: true });
    data.append(FP8x23 { mag: 4358996, sign: false });
    data.append(FP8x23 { mag: 4504227, sign: true });
    data.append(FP8x23 { mag: 7651888, sign: false });
    data.append(FP8x23 { mag: 6461454, sign: false });
    data.append(FP8x23 { mag: 246577, sign: false });
    data.append(FP8x23 { mag: 3137682, sign: false });
    data.append(FP8x23 { mag: 7875680, sign: false });
    data.append(FP8x23 { mag: 6510990, sign: false });
    data.append(FP8x23 { mag: 5786141, sign: true });
    data.append(FP8x23 { mag: 12351193, sign: true });
    data.append(FP8x23 { mag: 3913578, sign: false });
    data.append(FP8x23 { mag: 1045939, sign: false });
    data.append(FP8x23 { mag: 912629, sign: true });
    data.append(FP8x23 { mag: 528486, sign: false });
    data.append(FP8x23 { mag: 5421901, sign: true });
    data.append(FP8x23 { mag: 3900354, sign: true });
    data.append(FP8x23 { mag: 11383287, sign: true });
    data.append(FP8x23 { mag: 1493012, sign: true });
    data.append(FP8x23 { mag: 3276732, sign: false });
    data.append(FP8x23 { mag: 2835657, sign: false });
    data.append(FP8x23 { mag: 3872509, sign: false });
    data.append(FP8x23 { mag: 6404842, sign: false });
    data.append(FP8x23 { mag: 10064351, sign: true });
    data.append(FP8x23 { mag: 3261080, sign: false });
    data.append(FP8x23 { mag: 5934685, sign: false });
    data.append(FP8x23 { mag: 2819516, sign: false });
    data.append(FP8x23 { mag: 10171718, sign: false });
    data.append(FP8x23 { mag: 1165104, sign: false });
    data.append(FP8x23 { mag: 3961127, sign: true });
    data.append(FP8x23 { mag: 11299877, sign: false });
    data.append(FP8x23 { mag: 4346995, sign: false });
    data.append(FP8x23 { mag: 6139233, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
