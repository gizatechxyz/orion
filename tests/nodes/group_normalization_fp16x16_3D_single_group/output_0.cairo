use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 34275, sign: false });
    data.append(FP16x16 { mag: 5259, sign: true });
    data.append(FP16x16 { mag: 9950, sign: true });
    data.append(FP16x16 { mag: 66875, sign: true });
    data.append(FP16x16 { mag: 41059, sign: true });
    data.append(FP16x16 { mag: 97180, sign: false });
    data.append(FP16x16 { mag: 32094, sign: true });
    data.append(FP16x16 { mag: 32147, sign: true });
    data.append(FP16x16 { mag: 70679, sign: false });
    data.append(FP16x16 { mag: 2127, sign: false });
    data.append(FP16x16 { mag: 4360, sign: true });
    data.append(FP16x16 { mag: 52133, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
