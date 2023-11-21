use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 7, sign: false });
    data.append(i32 { mag: 11, sign: false });
    data.append(i32 { mag: 15, sign: false });
    data.append(i32 { mag: 19, sign: false });
    data.append(i32 { mag: 23, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
