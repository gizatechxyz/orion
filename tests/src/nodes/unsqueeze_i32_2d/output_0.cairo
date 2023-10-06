use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(2);
    shape.append(4);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 107, sign: true });
    data.append(i32 { mag: 114, sign: true });
    data.append(i32 { mag: 57, sign: false });
    data.append(i32 { mag: 126, sign: false });
    data.append(i32 { mag: 75, sign: false });
    data.append(i32 { mag: 121, sign: true });
    data.append(i32 { mag: 69, sign: false });
    data.append(i32 { mag: 117, sign: true });
    TensorTrait::new(shape.span(), data.span())
}