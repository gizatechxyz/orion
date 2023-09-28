use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 79, sign: true });
    data.append(i32 { mag: 43, sign: true });
    data.append(i32 { mag: 56, sign: false });
    data.append(i32 { mag: 124, sign: true });
    data.append(i32 { mag: 98, sign: false });
    data.append(i32 { mag: 123, sign: true });
    data.append(i32 { mag: 124, sign: false });
    data.append(i32 { mag: 86, sign: true });
    TensorTrait::new(shape.span(), data.span())
}