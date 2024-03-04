use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 53067, sign: false });
    data.append(FP16x16 { mag: 12598, sign: false });
    data.append(FP16x16 { mag: 22071, sign: true });
    data.append(FP16x16 { mag: 185160, sign: true });
    data.append(FP16x16 { mag: 71946, sign: true });
    data.append(FP16x16 { mag: 109757, sign: false });
    data.append(FP16x16 { mag: 53829, sign: true });
    data.append(FP16x16 { mag: 61215, sign: true });
    data.append(FP16x16 { mag: 112863, sign: false });
    data.append(FP16x16 { mag: 64576, sign: true });
    data.append(FP16x16 { mag: 14400, sign: true });
    data.append(FP16x16 { mag: 47376, sign: false });
    data.append(FP16x16 { mag: 42132, sign: false });
    data.append(FP16x16 { mag: 15521, sign: true });
    data.append(FP16x16 { mag: 19564, sign: false });
    data.append(FP16x16 { mag: 17481, sign: true });
    data.append(FP16x16 { mag: 1740, sign: true });
    data.append(FP16x16 { mag: 111657, sign: false });
    data.append(FP16x16 { mag: 112934, sign: false });
    data.append(FP16x16 { mag: 19870, sign: true });
    data.append(FP16x16 { mag: 122950, sign: true });
    data.append(FP16x16 { mag: 92754, sign: false });
    data.append(FP16x16 { mag: 3247, sign: false });
    data.append(FP16x16 { mag: 74346, sign: true });
    data.append(FP16x16 { mag: 98767, sign: false });
    data.append(FP16x16 { mag: 8702, sign: true });
    data.append(FP16x16 { mag: 40643, sign: true });
    data.append(FP16x16 { mag: 138135, sign: true });
    data.append(FP16x16 { mag: 22240, sign: false });
    data.append(FP16x16 { mag: 1595, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
