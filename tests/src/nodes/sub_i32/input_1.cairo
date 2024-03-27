use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_1() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 3, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
