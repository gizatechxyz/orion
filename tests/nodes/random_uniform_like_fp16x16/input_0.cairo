use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 70016, sign: true });
    data.append(FP16x16 { mag: 57536, sign: false });
    data.append(FP16x16 { mag: 116032, sign: false });
    data.append(FP16x16 { mag: 162944, sign: true });
    data.append(FP16x16 { mag: 43360, sign: false });
    data.append(FP16x16 { mag: 128960, sign: false });
    data.append(FP16x16 { mag: 151808, sign: true });
    data.append(FP16x16 { mag: 28368, sign: false });
    data.append(FP16x16 { mag: 21024, sign: false });
    data.append(FP16x16 { mag: 24992, sign: false });
    data.append(FP16x16 { mag: 125120, sign: true });
    data.append(FP16x16 { mag: 79168, sign: true });
    data.append(FP16x16 { mag: 136960, sign: true });
    data.append(FP16x16 { mag: 10104, sign: true });
    data.append(FP16x16 { mag: 136704, sign: false });
    data.append(FP16x16 { mag: 184960, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
