use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 156971, sign: false });
    data.append(FP16x16 { mag: 248406, sign: false });
    data.append(FP16x16 { mag: 431277, sign: false });
    data.append(FP16x16 { mag: 522712, sign: false });
    data.append(FP16x16 { mag: 614147, sign: false });
    data.append(FP16x16 { mag: 797018, sign: false });
    data.append(FP16x16 { mag: 888453, sign: false });
    data.append(FP16x16 { mag: 979888, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
