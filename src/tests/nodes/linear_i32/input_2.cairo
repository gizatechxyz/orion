use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I32Tensor;
use orion::numbers::{i32, FP16x16};

fn input_2() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 1, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}