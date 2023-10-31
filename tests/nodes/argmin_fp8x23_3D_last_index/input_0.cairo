use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 713031680, sign: false });
    data.append(FP8x23 { mag: 268435456, sign: false });
    data.append(FP8x23 { mag: 654311424, sign: false });
    data.append(FP8x23 { mag: 805306368, sign: true });
    data.append(FP8x23 { mag: 486539264, sign: true });
    data.append(FP8x23 { mag: 587202560, sign: true });
    data.append(FP8x23 { mag: 411041792, sign: true });
    data.append(FP8x23 { mag: 58720256, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
