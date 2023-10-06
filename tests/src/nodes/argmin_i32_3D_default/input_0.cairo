use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 28, sign: false });
    data.append(i32 { mag: 13, sign: false });
    data.append(i32 { mag: 50, sign: false });
    data.append(i32 { mag: 33, sign: true });
    data.append(i32 { mag: 64, sign: true });
    data.append(i32 { mag: 9, sign: true });
    data.append(i32 { mag: 55, sign: true });
    data.append(i32 { mag: 108, sign: false });
    TensorTrait::new(shape.span(), data.span())
}