use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 133576, sign: true });
    data.append(FP16x16 { mag: 63223, sign: false });
    data.append(FP16x16 { mag: 107763, sign: true });
    data.append(FP16x16 { mag: 13716, sign: true });
    data.append(FP16x16 { mag: 103890, sign: true });
    data.append(FP16x16 { mag: 101915, sign: true });
    data.append(FP16x16 { mag: 98400, sign: true });
    data.append(FP16x16 { mag: 4851, sign: false });
    data.append(FP16x16 { mag: 68295, sign: true });
    data.append(FP16x16 { mag: 177139, sign: true });
    data.append(FP16x16 { mag: 69916, sign: true });
    data.append(FP16x16 { mag: 8677, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
