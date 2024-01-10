use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 44676, sign: false });
    data.append(FP16x16 { mag: 53361, sign: true });
    data.append(FP16x16 { mag: 33378, sign: true });
    data.append(FP16x16 { mag: 43061, sign: false });
    data.append(FP16x16 { mag: 24801, sign: false });
    data.append(FP16x16 { mag: 33406, sign: false });
    data.append(FP16x16 { mag: 54529, sign: true });
    data.append(FP16x16 { mag: 133687, sign: false });
    data.append(FP16x16 { mag: 44032, sign: true });
    data.append(FP16x16 { mag: 38747, sign: true });
    data.append(FP16x16 { mag: 34054, sign: true });
    data.append(FP16x16 { mag: 45436, sign: false });
    data.append(FP16x16 { mag: 80815, sign: false });
    data.append(FP16x16 { mag: 79372, sign: true });
    data.append(FP16x16 { mag: 17958, sign: true });
    data.append(FP16x16 { mag: 49483, sign: false });
    data.append(FP16x16 { mag: 46695, sign: true });
    data.append(FP16x16 { mag: 1816, sign: true });
    data.append(FP16x16 { mag: 43264, sign: true });
    data.append(FP16x16 { mag: 59187, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
