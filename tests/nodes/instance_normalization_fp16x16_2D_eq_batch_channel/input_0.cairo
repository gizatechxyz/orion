use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 26118, sign: false });
    data.append(FP16x16 { mag: 105279, sign: true });
    data.append(FP16x16 { mag: 46181, sign: true });
    data.append(FP16x16 { mag: 87446, sign: true });
    data.append(FP16x16 { mag: 100793, sign: true });
    data.append(FP16x16 { mag: 38742, sign: true });
    data.append(FP16x16 { mag: 92288, sign: false });
    data.append(FP16x16 { mag: 12953, sign: true });
    data.append(FP16x16 { mag: 35879, sign: false });
    data.append(FP16x16 { mag: 50306, sign: false });
    data.append(FP16x16 { mag: 16435, sign: true });
    data.append(FP16x16 { mag: 88971, sign: false });
    data.append(FP16x16 { mag: 53396, sign: true });
    data.append(FP16x16 { mag: 115527, sign: true });
    data.append(FP16x16 { mag: 6641, sign: false });
    data.append(FP16x16 { mag: 107852, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
