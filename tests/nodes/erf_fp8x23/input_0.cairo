use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3776989, sign: true });
    data.append(FP8x23 { mag: 29442512, sign: true });
    data.append(FP8x23 { mag: 37326415, sign: false });
    data.append(FP8x23 { mag: 21784388, sign: true });
    data.append(FP8x23 { mag: 14455860, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
