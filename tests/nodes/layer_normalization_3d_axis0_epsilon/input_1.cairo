use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 53884, sign: false });
    data.append(FP16x16 { mag: 80807, sign: true });
    data.append(FP16x16 { mag: 17881, sign: true });
    data.append(FP16x16 { mag: 18476, sign: true });
    data.append(FP16x16 { mag: 36283, sign: true });
    data.append(FP16x16 { mag: 61330, sign: true });
    data.append(FP16x16 { mag: 41039, sign: false });
    data.append(FP16x16 { mag: 82065, sign: false });
    data.append(FP16x16 { mag: 45401, sign: false });
    data.append(FP16x16 { mag: 128118, sign: true });
    data.append(FP16x16 { mag: 214898, sign: true });
    data.append(FP16x16 { mag: 16418, sign: false });
    data.append(FP16x16 { mag: 82143, sign: false });
    data.append(FP16x16 { mag: 573, sign: true });
    data.append(FP16x16 { mag: 48898, sign: false });
    data.append(FP16x16 { mag: 14511, sign: false });
    data.append(FP16x16 { mag: 11366, sign: false });
    data.append(FP16x16 { mag: 53881, sign: false });
    data.append(FP16x16 { mag: 27317, sign: true });
    data.append(FP16x16 { mag: 88557, sign: false });
    data.append(FP16x16 { mag: 14203, sign: false });
    data.append(FP16x16 { mag: 1404, sign: true });
    data.append(FP16x16 { mag: 30266, sign: false });
    data.append(FP16x16 { mag: 83574, sign: true });
    data.append(FP16x16 { mag: 82692, sign: false });
    data.append(FP16x16 { mag: 86496, sign: false });
    data.append(FP16x16 { mag: 101363, sign: true });
    data.append(FP16x16 { mag: 30107, sign: true });
    data.append(FP16x16 { mag: 40283, sign: true });
    data.append(FP16x16 { mag: 54260, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
