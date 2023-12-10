use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 234881024, sign: false });
    data.append(FP8x23 { mag: 327155712, sign: false });
    data.append(FP8x23 { mag: 100663296, sign: true });
    data.append(FP8x23 { mag: 947912704, sign: true });
    data.append(FP8x23 { mag: 687865856, sign: true });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 998244352, sign: true });
    data.append(FP8x23 { mag: 805306368, sign: false });
    data.append(FP8x23 { mag: 855638016, sign: false });
    data.append(FP8x23 { mag: 1065353216, sign: true });
    data.append(FP8x23 { mag: 494927872, sign: false });
    data.append(FP8x23 { mag: 394264576, sign: false });
    data.append(FP8x23 { mag: 83886080, sign: false });
    data.append(FP8x23 { mag: 385875968, sign: true });
    data.append(FP8x23 { mag: 855638016, sign: true });
    data.append(FP8x23 { mag: 654311424, sign: true });
    data.append(FP8x23 { mag: 754974720, sign: true });
    data.append(FP8x23 { mag: 1023410176, sign: false });
    data.append(FP8x23 { mag: 520093696, sign: false });
    data.append(FP8x23 { mag: 654311424, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
