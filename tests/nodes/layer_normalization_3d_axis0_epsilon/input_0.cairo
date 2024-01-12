use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 73220, sign: false });
    data.append(FP16x16 { mag: 15621, sign: false });
    data.append(FP16x16 { mag: 26862, sign: true });
    data.append(FP16x16 { mag: 63971, sign: false });
    data.append(FP16x16 { mag: 64826, sign: false });
    data.append(FP16x16 { mag: 18837, sign: false });
    data.append(FP16x16 { mag: 66021, sign: true });
    data.append(FP16x16 { mag: 42181, sign: true });
    data.append(FP16x16 { mag: 69342, sign: true });
    data.append(FP16x16 { mag: 72001, sign: true });
    data.append(FP16x16 { mag: 99818, sign: true });
    data.append(FP16x16 { mag: 63088, sign: false });
    data.append(FP16x16 { mag: 17845, sign: true });
    data.append(FP16x16 { mag: 37020, sign: true });
    data.append(FP16x16 { mag: 20567, sign: false });
    data.append(FP16x16 { mag: 1924, sign: true });
    data.append(FP16x16 { mag: 13154, sign: true });
    data.append(FP16x16 { mag: 88735, sign: false });
    data.append(FP16x16 { mag: 40464, sign: false });
    data.append(FP16x16 { mag: 96907, sign: false });
    data.append(FP16x16 { mag: 79699, sign: false });
    data.append(FP16x16 { mag: 91862, sign: true });
    data.append(FP16x16 { mag: 97396, sign: false });
    data.append(FP16x16 { mag: 23929, sign: false });
    data.append(FP16x16 { mag: 11785, sign: true });
    data.append(FP16x16 { mag: 7747, sign: false });
    data.append(FP16x16 { mag: 91889, sign: true });
    data.append(FP16x16 { mag: 16735, sign: true });
    data.append(FP16x16 { mag: 120303, sign: true });
    data.append(FP16x16 { mag: 116144, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
