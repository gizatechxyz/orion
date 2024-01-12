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
    data.append(FP16x16 { mag: 51329, sign: true });
    data.append(FP16x16 { mag: 47264, sign: true });
    data.append(FP16x16 { mag: 78049, sign: false });
    data.append(FP16x16 { mag: 31332, sign: true });
    data.append(FP16x16 { mag: 64228, sign: false });
    data.append(FP16x16 { mag: 50183, sign: false });
    data.append(FP16x16 { mag: 111933, sign: true });
    data.append(FP16x16 { mag: 37549, sign: true });
    data.append(FP16x16 { mag: 48542, sign: true });
    data.append(FP16x16 { mag: 13252, sign: true });
    data.append(FP16x16 { mag: 63185, sign: true });
    data.append(FP16x16 { mag: 2871, sign: false });
    data.append(FP16x16 { mag: 57251, sign: false });
    data.append(FP16x16 { mag: 15125, sign: true });
    data.append(FP16x16 { mag: 75974, sign: false });
    data.append(FP16x16 { mag: 29448, sign: true });
    data.append(FP16x16 { mag: 118787, sign: false });
    data.append(FP16x16 { mag: 85238, sign: true });
    data.append(FP16x16 { mag: 6392, sign: true });
    data.append(FP16x16 { mag: 32667, sign: true });
    data.append(FP16x16 { mag: 306, sign: false });
    data.append(FP16x16 { mag: 53902, sign: true });
    data.append(FP16x16 { mag: 25940, sign: true });
    data.append(FP16x16 { mag: 38753, sign: true });
    data.append(FP16x16 { mag: 73289, sign: true });
    data.append(FP16x16 { mag: 47552, sign: false });
    data.append(FP16x16 { mag: 27826, sign: false });
    data.append(FP16x16 { mag: 47550, sign: false });
    data.append(FP16x16 { mag: 36199, sign: true });
    data.append(FP16x16 { mag: 43172, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
