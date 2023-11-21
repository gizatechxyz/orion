use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 110, sign: true });
    data.append(i32 { mag: 51, sign: false });
    data.append(i32 { mag: 114, sign: true });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 67, sign: true });
    data.append(i32 { mag: 10, sign: true });
    data.append(i32 { mag: 53, sign: true });
    data.append(i32 { mag: 58, sign: true });
    data.append(i32 { mag: 71, sign: false });
    data.append(i32 { mag: 36, sign: false });
    data.append(i32 { mag: 114, sign: false });
    data.append(i32 { mag: 76, sign: false });
    data.append(i32 { mag: 58, sign: false });
    data.append(i32 { mag: 98, sign: true });
    data.append(i32 { mag: 69, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
