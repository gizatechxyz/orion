use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 20971520, sign: false });
    data.append(FP16x16 { mag: 22020096, sign: false });
    data.append(FP16x16 { mag: 24117248, sign: false });
    data.append(FP16x16 { mag: 25165824, sign: false });
    data.append(FP16x16 { mag: 30408704, sign: false });
    data.append(FP16x16 { mag: 31457280, sign: false });
    data.append(FP16x16 { mag: 33554432, sign: false });
    data.append(FP16x16 { mag: 34603008, sign: false });
    data.append(FP16x16 { mag: 49283072, sign: false });
    data.append(FP16x16 { mag: 50331648, sign: false });
    data.append(FP16x16 { mag: 52428800, sign: false });
    data.append(FP16x16 { mag: 53477376, sign: false });
    data.append(FP16x16 { mag: 58720256, sign: false });
    data.append(FP16x16 { mag: 59768832, sign: false });
    data.append(FP16x16 { mag: 61865984, sign: false });
    data.append(FP16x16 { mag: 62914560, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
