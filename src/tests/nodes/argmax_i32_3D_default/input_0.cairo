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
    data.append(i32 { mag: 39, sign: true });
    data.append(i32 { mag: 30, sign: false });
    data.append(i32 { mag: 65, sign: false });
    data.append(i32 { mag: 96, sign: false });
    data.append(i32 { mag: 124, sign: false });
    data.append(i32 { mag: 63, sign: true });
    data.append(i32 { mag: 95, sign: false });
    data.append(i32 { mag: 2, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}