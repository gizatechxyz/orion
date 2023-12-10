use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 54, sign: false });
    data.append(FP8x23 { mag: 55, sign: false });
    data.append(FP8x23 { mag: 56, sign: false });
    data.append(FP8x23 { mag: 57, sign: false });
    data.append(FP8x23 { mag: 58, sign: false });
    data.append(FP8x23 { mag: 59, sign: false });
    data.append(FP8x23 { mag: 60, sign: false });
    data.append(FP8x23 { mag: 61, sign: false });
    data.append(FP8x23 { mag: 62, sign: false });
    data.append(FP8x23 { mag: 63, sign: false });
    data.append(FP8x23 { mag: 64, sign: false });
    data.append(FP8x23 { mag: 65, sign: false });
    data.append(FP8x23 { mag: 66, sign: false });
    data.append(FP8x23 { mag: 67, sign: false });
    data.append(FP8x23 { mag: 68, sign: false });
    data.append(FP8x23 { mag: 69, sign: false });
    data.append(FP8x23 { mag: 70, sign: false });
    data.append(FP8x23 { mag: 71, sign: false });
    data.append(FP8x23 { mag: 72, sign: false });
    data.append(FP8x23 { mag: 73, sign: false });
    data.append(FP8x23 { mag: 74, sign: false });
    data.append(FP8x23 { mag: 75, sign: false });
    data.append(FP8x23 { mag: 76, sign: false });
    data.append(FP8x23 { mag: 77, sign: false });
    data.append(FP8x23 { mag: 78, sign: false });
    data.append(FP8x23 { mag: 79, sign: false });
    data.append(FP8x23 { mag: 80, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
