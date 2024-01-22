use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 13522, sign: false });
    data.append(FP16x16 { mag: 12888, sign: false });
    data.append(FP16x16 { mag: 53048, sign: false });
    data.append(FP16x16 { mag: 29185, sign: false });
    data.append(FP16x16 { mag: 17013, sign: false });
    data.append(FP16x16 { mag: 54418, sign: false });
    data.append(FP16x16 { mag: 48187, sign: false });
    data.append(FP16x16 { mag: 45789, sign: false });
    data.append(FP16x16 { mag: 20854, sign: false });
    data.append(FP16x16 { mag: 49110, sign: false });
    data.append(FP16x16 { mag: 48981, sign: false });
    data.append(FP16x16 { mag: 9983, sign: false });
    data.append(FP16x16 { mag: 4283, sign: false });
    data.append(FP16x16 { mag: 14740, sign: false });
    data.append(FP16x16 { mag: 39006, sign: false });
    data.append(FP16x16 { mag: 38039, sign: false });
    data.append(FP16x16 { mag: 39666, sign: false });
    data.append(FP16x16 { mag: 3808, sign: false });
    data.append(FP16x16 { mag: 34989, sign: false });
    data.append(FP16x16 { mag: 49693, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
