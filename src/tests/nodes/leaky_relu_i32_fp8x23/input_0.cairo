use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I32Tensor;
use orion::numbers::{i32, FP8x23};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 7, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 2, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}