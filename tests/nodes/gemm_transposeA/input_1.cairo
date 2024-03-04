use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 38056, sign: false });
    data.append(FP16x16 { mag: 35209, sign: false });
    data.append(FP16x16 { mag: 55369, sign: false });
    data.append(FP16x16 { mag: 64248, sign: false });
    data.append(FP16x16 { mag: 16398, sign: false });
    data.append(FP16x16 { mag: 58129, sign: false });
    data.append(FP16x16 { mag: 46239, sign: false });
    data.append(FP16x16 { mag: 56435, sign: false });
    data.append(FP16x16 { mag: 25984, sign: false });
    data.append(FP16x16 { mag: 64809, sign: false });
    data.append(FP16x16 { mag: 65020, sign: false });
    data.append(FP16x16 { mag: 8759, sign: false });
    data.append(FP16x16 { mag: 50946, sign: false });
    data.append(FP16x16 { mag: 29640, sign: false });
    data.append(FP16x16 { mag: 12257, sign: false });
    data.append(FP16x16 { mag: 26776, sign: false });
    data.append(FP16x16 { mag: 20618, sign: false });
    data.append(FP16x16 { mag: 15242, sign: false });
    data.append(FP16x16 { mag: 5560, sign: false });
    data.append(FP16x16 { mag: 3669, sign: false });
    data.append(FP16x16 { mag: 64019, sign: false });
    data.append(FP16x16 { mag: 9851, sign: false });
    data.append(FP16x16 { mag: 26388, sign: false });
    data.append(FP16x16 { mag: 62343, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
