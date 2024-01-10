use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 93757, sign: false });
    data.append(FP16x16 { mag: 4959, sign: true });
    data.append(FP16x16 { mag: 26505, sign: false });
    data.append(FP16x16 { mag: 1530, sign: false });
    data.append(FP16x16 { mag: 75165, sign: true });
    data.append(FP16x16 { mag: 97790, sign: true });
    data.append(FP16x16 { mag: 16969, sign: true });
    data.append(FP16x16 { mag: 61662, sign: false });
    data.append(FP16x16 { mag: 110907, sign: true });
    data.append(FP16x16 { mag: 17227, sign: true });
    data.append(FP16x16 { mag: 14091, sign: false });
    data.append(FP16x16 { mag: 52957, sign: true });
    data.append(FP16x16 { mag: 41342, sign: false });
    data.append(FP16x16 { mag: 34186, sign: false });
    data.append(FP16x16 { mag: 17811, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
