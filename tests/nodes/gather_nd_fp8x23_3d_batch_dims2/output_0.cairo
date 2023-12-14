use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16777216, sign: false });
    data.append(FP8x23 { mag: 58720256, sign: false });
    data.append(FP8x23 { mag: 125829120, sign: false });
    data.append(FP8x23 { mag: 167772160, sign: false });
    data.append(FP8x23 { mag: 226492416, sign: false });
    data.append(FP8x23 { mag: 268435456, sign: false });
    data.append(FP8x23 { mag: 301989888, sign: false });
    data.append(FP8x23 { mag: 360710144, sign: false });
    data.append(FP8x23 { mag: 411041792, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
