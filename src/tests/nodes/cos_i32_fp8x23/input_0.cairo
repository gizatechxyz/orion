use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};

use orion::operators::tensor::implementations::tensor_i32_fp8x23::Tensor_i32_fp8x23;
use orion::numbers::{i32, FP8x23};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 110, sign: false });
    data.append(i32 { mag: 61, sign: false });
    data.append(i32 { mag: 124, sign: false });
    data.append(i32 { mag: 95, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}