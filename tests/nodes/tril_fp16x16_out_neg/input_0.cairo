use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 3538944, sign: true });
    data.append(FP16x16 { mag: 720896, sign: false });
    data.append(FP16x16 { mag: 3473408, sign: true });
    data.append(FP16x16 { mag: 6422528, sign: true });
    data.append(FP16x16 { mag: 1703936, sign: true });
    data.append(FP16x16 { mag: 1507328, sign: false });
    data.append(FP16x16 { mag: 4325376, sign: false });
    data.append(FP16x16 { mag: 6619136, sign: true });
    data.append(FP16x16 { mag: 3342336, sign: true });
    data.append(FP16x16 { mag: 5111808, sign: false });
    data.append(FP16x16 { mag: 3473408, sign: false });
    data.append(FP16x16 { mag: 6881280, sign: true });
    data.append(FP16x16 { mag: 917504, sign: false });
    data.append(FP16x16 { mag: 3407872, sign: true });
    data.append(FP16x16 { mag: 6488064, sign: false });
    data.append(FP16x16 { mag: 5636096, sign: false });
    data.append(FP16x16 { mag: 4521984, sign: true });
    data.append(FP16x16 { mag: 7077888, sign: true });
    data.append(FP16x16 { mag: 2359296, sign: false });
    data.append(FP16x16 { mag: 2097152, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
