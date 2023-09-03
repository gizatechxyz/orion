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
    data.append(i32 { mag: 78, sign: false });
    data.append(i32 { mag: 52, sign: false });
    data.append(i32 { mag: 26, sign: true });
    data.append(i32 { mag: 56, sign: false });
    data.append(i32 { mag: 122, sign: false });
    data.append(i32 { mag: 68, sign: true });
    data.append(i32 { mag: 41, sign: true });
    data.append(i32 { mag: 32, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}