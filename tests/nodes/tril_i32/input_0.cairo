use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 85, sign: false });
    data.append(i32 { mag: 107, sign: false });
    data.append(i32 { mag: 31, sign: true });
    data.append(i32 { mag: 49, sign: false });
    data.append(i32 { mag: 31, sign: true });
    data.append(i32 { mag: 9, sign: true });
    data.append(i32 { mag: 82, sign: true });
    data.append(i32 { mag: 11, sign: false });
    data.append(i32 { mag: 4, sign: false });
    data.append(i32 { mag: 64, sign: true });
    data.append(i32 { mag: 62, sign: true });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 98, sign: false });
    data.append(i32 { mag: 96, sign: false });
    data.append(i32 { mag: 116, sign: false });
    data.append(i32 { mag: 78, sign: false });
    data.append(i32 { mag: 111, sign: false });
    data.append(i32 { mag: 98, sign: true });
    data.append(i32 { mag: 34, sign: false });
    data.append(i32 { mag: 51, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
