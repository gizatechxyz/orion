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
    data.append(FP16x16 { mag: 93487, sign: true });
    data.append(FP16x16 { mag: 10869, sign: true });
    data.append(FP16x16 { mag: 157163, sign: false });
    data.append(FP16x16 { mag: 42999, sign: true });
    data.append(FP16x16 { mag: 34573, sign: true });
    data.append(FP16x16 { mag: 69783, sign: true });
    data.append(FP16x16 { mag: 171294, sign: false });
    data.append(FP16x16 { mag: 57130, sign: true });
    data.append(FP16x16 { mag: 11656, sign: true });
    data.append(FP16x16 { mag: 92700, sign: true });
    data.append(FP16x16 { mag: 28600, sign: true });
    data.append(FP16x16 { mag: 142763, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
