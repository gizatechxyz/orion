use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 26141, sign: false });
    data.append(FP16x16 { mag: 65210, sign: false });
    data.append(FP16x16 { mag: 44238, sign: false });
    data.append(FP16x16 { mag: 15740, sign: false });
    data.append(FP16x16 { mag: 27732, sign: false });
    data.append(FP16x16 { mag: 54749, sign: false });
    data.append(FP16x16 { mag: 23531, sign: false });
    data.append(FP16x16 { mag: 12333, sign: false });
    data.append(FP16x16 { mag: 3048, sign: false });
    data.append(FP16x16 { mag: 50642, sign: false });
    data.append(FP16x16 { mag: 50567, sign: false });
    data.append(FP16x16 { mag: 20553, sign: false });
    data.append(FP16x16 { mag: 14789, sign: false });
    data.append(FP16x16 { mag: 48807, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
