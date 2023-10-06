use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 83886080, sign: true });
    data.append(FP8x23 { mag: 117440512, sign: false });
    data.append(FP8x23 { mag: 25165824, sign: false });
    data.append(FP8x23 { mag: 662700032, sign: false });
    data.append(FP8x23 { mag: 285212672, sign: false });
    data.append(FP8x23 { mag: 260046848, sign: false });
    data.append(FP8x23 { mag: 469762048, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: false });
    TensorTrait::new(shape.span(), data.span())
}