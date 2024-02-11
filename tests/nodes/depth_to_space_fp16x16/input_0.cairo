use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 172672, sign: false });
    data.append(FP16x16 { mag: 110976, sign: true });
    data.append(FP16x16 { mag: 102912, sign: true });
    data.append(FP16x16 { mag: 146944, sign: true });
    data.append(FP16x16 { mag: 159232, sign: false });
    data.append(FP16x16 { mag: 130112, sign: false });
    data.append(FP16x16 { mag: 106304, sign: false });
    data.append(FP16x16 { mag: 26832, sign: false });
    data.append(FP16x16 { mag: 26800, sign: false });
    data.append(FP16x16 { mag: 172928, sign: true });
    data.append(FP16x16 { mag: 177280, sign: true });
    data.append(FP16x16 { mag: 102208, sign: false });
    data.append(FP16x16 { mag: 11808, sign: true });
    data.append(FP16x16 { mag: 111488, sign: true });
    data.append(FP16x16 { mag: 53120, sign: true });
    data.append(FP16x16 { mag: 165888, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
