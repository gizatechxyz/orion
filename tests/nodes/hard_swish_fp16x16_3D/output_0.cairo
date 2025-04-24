use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 18559, sign: true });
    data.append(FP16x16 { mag: 14645, sign: true });
    data.append(FP16x16 { mag: 141626, sign: false });
    data.append(FP16x16 { mag: 23208, sign: true });
    data.append(FP16x16 { mag: 137163, sign: false });
    data.append(FP16x16 { mag: 158684, sign: false });
    data.append(FP16x16 { mag: 13704, sign: true });
    data.append(FP16x16 { mag: 24389, sign: true });
    data.append(FP16x16 { mag: 20673, sign: true });
    data.append(FP16x16 { mag: 133, sign: true });
    data.append(FP16x16 { mag: 11084, sign: false });
    data.append(FP16x16 { mag: 1714, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
