use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 77, sign: true });
    data.append(i32 { mag: 114, sign: false });
    data.append(i32 { mag: 57, sign: false });
    data.append(i32 { mag: 10, sign: false });
    data.append(i32 { mag: 16, sign: false });
    data.append(i32 { mag: 80, sign: false });
    data.append(i32 { mag: 49, sign: true });
    data.append(i32 { mag: 41, sign: false });
    data.append(i32 { mag: 77, sign: true });
    data.append(i32 { mag: 47, sign: false });
    data.append(i32 { mag: 27, sign: true });
    data.append(i32 { mag: 108, sign: false });
    data.append(i32 { mag: 59, sign: false });
    data.append(i32 { mag: 32, sign: false });
    data.append(i32 { mag: 110, sign: true });
    data.append(i32 { mag: 102, sign: false });
    data.append(i32 { mag: 104, sign: false });
    data.append(i32 { mag: 17, sign: true });
    data.append(i32 { mag: 110, sign: true });
    data.append(i32 { mag: 16, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
