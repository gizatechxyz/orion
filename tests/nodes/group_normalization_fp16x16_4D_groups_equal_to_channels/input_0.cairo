use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 12945, sign: true });
    data.append(FP16x16 { mag: 99443, sign: true });
    data.append(FP16x16 { mag: 41621, sign: true });
    data.append(FP16x16 { mag: 87799, sign: true });
    data.append(FP16x16 { mag: 41868, sign: true });
    data.append(FP16x16 { mag: 38196, sign: true });
    data.append(FP16x16 { mag: 60536, sign: false });
    data.append(FP16x16 { mag: 64171, sign: false });
    data.append(FP16x16 { mag: 12480, sign: false });
    data.append(FP16x16 { mag: 14201, sign: false });
    data.append(FP16x16 { mag: 3824, sign: true });
    data.append(FP16x16 { mag: 38638, sign: true });
    data.append(FP16x16 { mag: 82872, sign: false });
    data.append(FP16x16 { mag: 9268, sign: false });
    data.append(FP16x16 { mag: 77427, sign: true });
    data.append(FP16x16 { mag: 28499, sign: false });
    data.append(FP16x16 { mag: 8796, sign: true });
    data.append(FP16x16 { mag: 88902, sign: true });
    data.append(FP16x16 { mag: 91378, sign: false });
    data.append(FP16x16 { mag: 23232, sign: false });
    data.append(FP16x16 { mag: 7318, sign: false });
    data.append(FP16x16 { mag: 109962, sign: false });
    data.append(FP16x16 { mag: 62751, sign: true });
    data.append(FP16x16 { mag: 15561, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
