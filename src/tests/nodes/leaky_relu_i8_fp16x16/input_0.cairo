use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I8Tensor;
use orion::numbers::{i8, FP16x16};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 3, sign: true });
    data.append(i8 { mag: 4, sign: false });
    data.append(i8 { mag: 7, sign: false });
    data.append(i8 { mag: 2, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}