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
    data.append(FP16x16 { mag: 47175, sign: false });
    data.append(FP16x16 { mag: 48068, sign: false });
    data.append(FP16x16 { mag: 72715, sign: true });
    data.append(FP16x16 { mag: 40805, sign: true });
    data.append(FP16x16 { mag: 97387, sign: true });
    data.append(FP16x16 { mag: 52917, sign: false });
    data.append(FP16x16 { mag: 16842, sign: false });
    data.append(FP16x16 { mag: 48741, sign: false });
    data.append(FP16x16 { mag: 114144, sign: false });
    data.append(FP16x16 { mag: 5783, sign: false });
    data.append(FP16x16 { mag: 24320, sign: true });
    data.append(FP16x16 { mag: 1477, sign: true });
    data.append(FP16x16 { mag: 101470, sign: false });
    data.append(FP16x16 { mag: 126422, sign: true });
    data.append(FP16x16 { mag: 5733, sign: true });
    data.append(FP16x16 { mag: 97314, sign: true });
    data.append(FP16x16 { mag: 65247, sign: false });
    data.append(FP16x16 { mag: 3443, sign: true });
    data.append(FP16x16 { mag: 16494, sign: true });
    data.append(FP16x16 { mag: 107377, sign: true });
    data.append(FP16x16 { mag: 63019, sign: false });
    data.append(FP16x16 { mag: 2523, sign: false });
    data.append(FP16x16 { mag: 111859, sign: true });
    data.append(FP16x16 { mag: 26857, sign: true });
    data.append(FP16x16 { mag: 19691, sign: true });
    data.append(FP16x16 { mag: 83436, sign: true });
    data.append(FP16x16 { mag: 176674, sign: true });
    data.append(FP16x16 { mag: 21073, sign: true });
    data.append(FP16x16 { mag: 114640, sign: false });
    data.append(FP16x16 { mag: 26239, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
