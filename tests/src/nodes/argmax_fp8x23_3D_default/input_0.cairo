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
    data.append(FP8x23 { mag: 536870912, sign: false });
    data.append(FP8x23 { mag: 301989888, sign: true });
    data.append(FP8x23 { mag: 310378496, sign: true });
    data.append(FP8x23 { mag: 394264576, sign: true });
    data.append(FP8x23 { mag: 75497472, sign: true });
    data.append(FP8x23 { mag: 285212672, sign: true });
    data.append(FP8x23 { mag: 419430400, sign: false });
    data.append(FP8x23 { mag: 201326592, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
