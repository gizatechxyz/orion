use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 176052, sign: false });
    data.append(FP16x16 { mag: 161653, sign: false });
    data.append(FP16x16 { mag: 91548, sign: false });
    data.append(FP16x16 { mag: 130822, sign: false });
    data.append(FP16x16 { mag: 113090, sign: false });
    data.append(FP16x16 { mag: 114286, sign: false });
    data.append(FP16x16 { mag: 78771, sign: false });
    data.append(FP16x16 { mag: 55867, sign: false });
    data.append(FP16x16 { mag: 83763, sign: false });
    data.append(FP16x16 { mag: 103037, sign: false });
    data.append(FP16x16 { mag: 53197, sign: false });
    data.append(FP16x16 { mag: 82622, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
