use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 33076, sign: false });
    data.append(FP16x16 { mag: 49782, sign: false });
    data.append(FP16x16 { mag: 120002, sign: false });
    data.append(FP16x16 { mag: 100565, sign: false });
    data.append(FP16x16 { mag: 26262, sign: false });
    data.append(FP16x16 { mag: 67414, sign: false });
    data.append(FP16x16 { mag: 93642, sign: false });
    data.append(FP16x16 { mag: 90109, sign: false });
    data.append(FP16x16 { mag: 29452, sign: false });
    data.append(FP16x16 { mag: 70161, sign: false });
    data.append(FP16x16 { mag: 104534, sign: false });
    data.append(FP16x16 { mag: 59471, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
