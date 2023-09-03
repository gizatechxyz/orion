use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I32Tensor;
use orion::numbers::{i32, FP16x16};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 96, sign: false });
    data.append(i32 { mag: 98, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 15, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}