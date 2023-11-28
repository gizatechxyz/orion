use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 157414, sign: true });
    data.append(FP16x16 { mag: 1771, sign: true });
    data.append(FP16x16 { mag: 104423, sign: false });
    data.append(FP16x16 { mag: 2093, sign: true });
    data.append(FP16x16 { mag: 63936, sign: false });
    data.append(FP16x16 { mag: 120461, sign: false });
    data.append(FP16x16 { mag: 72730, sign: false });
    data.append(FP16x16 { mag: 15555, sign: true });
    data.append(FP16x16 { mag: 44984, sign: false });
    data.append(FP16x16 { mag: 94211, sign: false });
    data.append(FP16x16 { mag: 93317, sign: true });
    data.append(FP16x16 { mag: 29023, sign: true });
    data.append(FP16x16 { mag: 52455, sign: true });
    data.append(FP16x16 { mag: 176026, sign: true });
    data.append(FP16x16 { mag: 98593, sign: true });
    data.append(FP16x16 { mag: 164495, sign: true });
    data.append(FP16x16 { mag: 140493, sign: true });
    data.append(FP16x16 { mag: 125737, sign: true });
    data.append(FP16x16 { mag: 64810, sign: false });
    data.append(FP16x16 { mag: 89675, sign: true });
    data.append(FP16x16 { mag: 125404, sign: false });
    data.append(FP16x16 { mag: 89381, sign: true });
    data.append(FP16x16 { mag: 93376, sign: false });
    data.append(FP16x16 { mag: 26320, sign: true });
    data.append(FP16x16 { mag: 2108, sign: true });
    data.append(FP16x16 { mag: 132017, sign: false });
    data.append(FP16x16 { mag: 69573, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
