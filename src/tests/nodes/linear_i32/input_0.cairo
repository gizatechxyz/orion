use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I32Tensor;
use orion::numbers::{i32, FP16x16};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 8, sign: false });
    data.append(i32 { mag: 7, sign: false });
    data.append(i32 { mag: 5, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}