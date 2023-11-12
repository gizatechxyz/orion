use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 61, sign: true });
    data.append(i32 { mag: 38, sign: true });
    data.append(i32 { mag: 71, sign: false });
    data.append(i32 { mag: 43, sign: false });
    data.append(i32 { mag: 63, sign: true });
    data.append(i32 { mag: 107, sign: false });
    data.append(i32 { mag: 47, sign: false });
    data.append(i32 { mag: 74, sign: true });
    data.append(i32 { mag: 32, sign: true });
    data.append(i32 { mag: 51, sign: false });
    data.append(i32 { mag: 8, sign: true });
    data.append(i32 { mag: 5, sign: false });
    data.append(i32 { mag: 89, sign: true });
    data.append(i32 { mag: 110, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 93, sign: false });
    data.append(i32 { mag: 117, sign: false });
    data.append(i32 { mag: 65, sign: false });
    data.append(i32 { mag: 99, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
