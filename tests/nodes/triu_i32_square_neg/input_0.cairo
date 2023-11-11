use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 114, sign: false });
    data.append(i32 { mag: 120, sign: true });
    data.append(i32 { mag: 15, sign: false });
    data.append(i32 { mag: 90, sign: true });
    data.append(i32 { mag: 31, sign: false });
    data.append(i32 { mag: 109, sign: true });
    data.append(i32 { mag: 81, sign: false });
    data.append(i32 { mag: 42, sign: false });
    data.append(i32 { mag: 8, sign: true });
    data.append(i32 { mag: 19, sign: true });
    data.append(i32 { mag: 69, sign: true });
    data.append(i32 { mag: 49, sign: false });
    data.append(i32 { mag: 106, sign: false });
    data.append(i32 { mag: 29, sign: true });
    data.append(i32 { mag: 13, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 54, sign: true });
    data.append(i32 { mag: 121, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
