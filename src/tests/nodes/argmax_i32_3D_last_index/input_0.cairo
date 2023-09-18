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
    data.append(i32 { mag: 7, sign: true });
    data.append(i32 { mag: 88, sign: false });
    data.append(i32 { mag: 46, sign: false });
    data.append(i32 { mag: 72, sign: false });
    data.append(i32 { mag: 20, sign: false });
    data.append(i32 { mag: 87, sign: false });
    data.append(i32 { mag: 67, sign: true });
    data.append(i32 { mag: 110, sign: false });
    TensorTrait::new(shape.span(), data.span())
}