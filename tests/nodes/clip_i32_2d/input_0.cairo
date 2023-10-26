use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 100, sign: false });
    data.append(i32 { mag: 100, sign: false });
    data.append(i32 { mag: 15, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 62, sign: false });
    data.append(i32 { mag: 47, sign: true });
    data.append(i32 { mag: 37, sign: false });
    data.append(i32 { mag: 50, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
