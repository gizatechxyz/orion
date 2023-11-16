use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 19, sign: false });
    data.append(i32 { mag: 35, sign: false });
    data.append(i32 { mag: 61, sign: false });
    data.append(i32 { mag: 58, sign: true });
    data.append(i32 { mag: 91, sign: true });
    data.append(i32 { mag: 4, sign: false });
    data.append(i32 { mag: 40, sign: true });
    data.append(i32 { mag: 88, sign: false });
    data.append(i32 { mag: 86, sign: true });
    data.append(i32 { mag: 76, sign: true });
    data.append(i32 { mag: 96, sign: true });
    data.append(i32 { mag: 39, sign: true });
    data.append(i32 { mag: 100, sign: false });
    data.append(i32 { mag: 26, sign: true });
    data.append(i32 { mag: 18, sign: false });
    data.append(i32 { mag: 99, sign: false });
    data.append(i32 { mag: 62, sign: false });
    data.append(i32 { mag: 21, sign: false });
    data.append(i32 { mag: 93, sign: true });
    data.append(i32 { mag: 71, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
