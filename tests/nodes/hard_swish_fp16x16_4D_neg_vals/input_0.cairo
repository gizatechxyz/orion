use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 340045, sign: true });
    data.append(FP16x16 { mag: 351017, sign: true });
    data.append(FP16x16 { mag: 249035, sign: true });
    data.append(FP16x16 { mag: 221419, sign: true });
    data.append(FP16x16 { mag: 213842, sign: true });
    data.append(FP16x16 { mag: 382721, sign: true });
    data.append(FP16x16 { mag: 291006, sign: true });
    data.append(FP16x16 { mag: 385065, sign: true });
    data.append(FP16x16 { mag: 257560, sign: true });
    data.append(FP16x16 { mag: 332095, sign: true });
    data.append(FP16x16 { mag: 331475, sign: true });
    data.append(FP16x16 { mag: 381703, sign: true });
    data.append(FP16x16 { mag: 265151, sign: true });
    data.append(FP16x16 { mag: 255429, sign: true });
    data.append(FP16x16 { mag: 346011, sign: true });
    data.append(FP16x16 { mag: 203141, sign: true });
    data.append(FP16x16 { mag: 378342, sign: true });
    data.append(FP16x16 { mag: 356996, sign: true });
    data.append(FP16x16 { mag: 217915, sign: true });
    data.append(FP16x16 { mag: 317677, sign: true });
    data.append(FP16x16 { mag: 388705, sign: true });
    data.append(FP16x16 { mag: 225293, sign: true });
    data.append(FP16x16 { mag: 328602, sign: true });
    data.append(FP16x16 { mag: 285685, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
