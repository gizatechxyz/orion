use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 43014, sign: false });
    data.append(FP16x16 { mag: 16802, sign: false });
    data.append(FP16x16 { mag: 52318, sign: false });
    data.append(FP16x16 { mag: 40979, sign: false });
    data.append(FP16x16 { mag: 277, sign: false });
    data.append(FP16x16 { mag: 29333, sign: false });
    data.append(FP16x16 { mag: 26593, sign: false });
    data.append(FP16x16 { mag: 41294, sign: false });
    data.append(FP16x16 { mag: 47646, sign: false });
    data.append(FP16x16 { mag: 2148, sign: false });
    data.append(FP16x16 { mag: 24564, sign: false });
    data.append(FP16x16 { mag: 18370, sign: false });
    data.append(FP16x16 { mag: 63747, sign: false });
    data.append(FP16x16 { mag: 10833, sign: false });
    data.append(FP16x16 { mag: 29301, sign: false });
    data.append(FP16x16 { mag: 20991, sign: false });
    data.append(FP16x16 { mag: 28040, sign: false });
    data.append(FP16x16 { mag: 133, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
