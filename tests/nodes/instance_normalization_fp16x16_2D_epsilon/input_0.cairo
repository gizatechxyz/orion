use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 98908, sign: true });
    data.append(FP16x16 { mag: 12461, sign: false });
    data.append(FP16x16 { mag: 3163, sign: false });
    data.append(FP16x16 { mag: 6984, sign: false });
    data.append(FP16x16 { mag: 4140, sign: true });
    data.append(FP16x16 { mag: 47930, sign: false });
    data.append(FP16x16 { mag: 18732, sign: false });
    data.append(FP16x16 { mag: 72156, sign: false });
    data.append(FP16x16 { mag: 12344, sign: false });
    data.append(FP16x16 { mag: 70537, sign: true });
    data.append(FP16x16 { mag: 177184, sign: false });
    data.append(FP16x16 { mag: 23585, sign: true });
    data.append(FP16x16 { mag: 73750, sign: true });
    data.append(FP16x16 { mag: 28809, sign: false });
    data.append(FP16x16 { mag: 107918, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
