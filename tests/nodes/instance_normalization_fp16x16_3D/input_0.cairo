use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(2);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 24324, sign: true });
    data.append(FP16x16 { mag: 40415, sign: true });
    data.append(FP16x16 { mag: 69653, sign: false });
    data.append(FP16x16 { mag: 30839, sign: false });
    data.append(FP16x16 { mag: 61645, sign: false });
    data.append(FP16x16 { mag: 80064, sign: false });
    data.append(FP16x16 { mag: 65586, sign: true });
    data.append(FP16x16 { mag: 32775, sign: true });
    data.append(FP16x16 { mag: 46004, sign: true });
    data.append(FP16x16 { mag: 23206, sign: false });
    data.append(FP16x16 { mag: 15017, sign: false });
    data.append(FP16x16 { mag: 47416, sign: false });
    data.append(FP16x16 { mag: 6910, sign: true });
    data.append(FP16x16 { mag: 138625, sign: false });
    data.append(FP16x16 { mag: 119926, sign: false });
    data.append(FP16x16 { mag: 25243, sign: false });
    data.append(FP16x16 { mag: 110184, sign: false });
    data.append(FP16x16 { mag: 43054, sign: true });
    data.append(FP16x16 { mag: 116187, sign: true });
    data.append(FP16x16 { mag: 3371, sign: true });
    data.append(FP16x16 { mag: 71501, sign: true });
    data.append(FP16x16 { mag: 123235, sign: true });
    data.append(FP16x16 { mag: 23436, sign: false });
    data.append(FP16x16 { mag: 4438, sign: false });
    data.append(FP16x16 { mag: 44590, sign: false });
    data.append(FP16x16 { mag: 22327, sign: true });
    data.append(FP16x16 { mag: 46064, sign: false });
    data.append(FP16x16 { mag: 93234, sign: true });
    data.append(FP16x16 { mag: 5227, sign: false });
    data.append(FP16x16 { mag: 90602, sign: true });
    data.append(FP16x16 { mag: 85854, sign: false });
    data.append(FP16x16 { mag: 36001, sign: false });
    data.append(FP16x16 { mag: 77769, sign: false });
    data.append(FP16x16 { mag: 21352, sign: false });
    data.append(FP16x16 { mag: 115104, sign: true });
    data.append(FP16x16 { mag: 2269, sign: true });
    data.append(FP16x16 { mag: 85752, sign: false });
    data.append(FP16x16 { mag: 57450, sign: false });
    data.append(FP16x16 { mag: 48540, sign: false });
    data.append(FP16x16 { mag: 47360, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
