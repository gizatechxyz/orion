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
    data.append(FP16x16 { mag: 59178, sign: true });
    data.append(FP16x16 { mag: 9314, sign: false });
    data.append(FP16x16 { mag: 44617, sign: false });
    data.append(FP16x16 { mag: 36687, sign: false });
    data.append(FP16x16 { mag: 71557, sign: true });
    data.append(FP16x16 { mag: 75061, sign: false });
    data.append(FP16x16 { mag: 36819, sign: false });
    data.append(FP16x16 { mag: 34756, sign: false });
    data.append(FP16x16 { mag: 33152, sign: false });
    data.append(FP16x16 { mag: 46154, sign: false });
    data.append(FP16x16 { mag: 35825, sign: false });
    data.append(FP16x16 { mag: 21935, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
