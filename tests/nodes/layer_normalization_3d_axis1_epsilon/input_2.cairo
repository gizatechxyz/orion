use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 37055, sign: true });
    data.append(FP16x16 { mag: 26922, sign: false });
    data.append(FP16x16 { mag: 155904, sign: false });
    data.append(FP16x16 { mag: 33841, sign: true });
    data.append(FP16x16 { mag: 53256, sign: false });
    data.append(FP16x16 { mag: 22490, sign: false });
    data.append(FP16x16 { mag: 110070, sign: false });
    data.append(FP16x16 { mag: 90061, sign: true });
    data.append(FP16x16 { mag: 44130, sign: true });
    data.append(FP16x16 { mag: 8720, sign: true });
    data.append(FP16x16 { mag: 61513, sign: true });
    data.append(FP16x16 { mag: 42238, sign: true });
    data.append(FP16x16 { mag: 18154, sign: false });
    data.append(FP16x16 { mag: 88282, sign: false });
    data.append(FP16x16 { mag: 29231, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
