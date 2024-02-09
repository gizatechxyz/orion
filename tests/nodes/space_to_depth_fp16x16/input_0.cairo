use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 35648, sign: false });
    data.append(FP16x16 { mag: 93312, sign: false });
    data.append(FP16x16 { mag: 68608, sign: false });
    data.append(FP16x16 { mag: 93888, sign: true });
    data.append(FP16x16 { mag: 180864, sign: false });
    data.append(FP16x16 { mag: 7268, sign: false });
    data.append(FP16x16 { mag: 188800, sign: true });
    data.append(FP16x16 { mag: 104576, sign: true });
    data.append(FP16x16 { mag: 84288, sign: true });
    data.append(FP16x16 { mag: 44864, sign: false });
    data.append(FP16x16 { mag: 180480, sign: false });
    data.append(FP16x16 { mag: 147584, sign: true });
    data.append(FP16x16 { mag: 179584, sign: true });
    data.append(FP16x16 { mag: 172800, sign: false });
    data.append(FP16x16 { mag: 182912, sign: true });
    data.append(FP16x16 { mag: 19408, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
