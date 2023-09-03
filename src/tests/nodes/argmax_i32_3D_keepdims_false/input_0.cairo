use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I32Tensor;
use orion::numbers::{i32, FP16x16};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 100, sign: true });
    data.append(i32 { mag: 96, sign: false });
    data.append(i32 { mag: 5, sign: false });
    data.append(i32 { mag: 30, sign: true });
    data.append(i32 { mag: 13, sign: false });
    data.append(i32 { mag: 123, sign: true });
    data.append(i32 { mag: 60, sign: true });
    data.append(i32 { mag: 59, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}