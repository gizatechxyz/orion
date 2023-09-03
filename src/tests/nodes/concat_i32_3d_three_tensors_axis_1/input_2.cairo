use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I32Tensor;
use orion::numbers::{i32, FP16x16};

fn input_2() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 54, sign: false });
    data.append(i32 { mag: 55, sign: false });
    data.append(i32 { mag: 56, sign: false });
    data.append(i32 { mag: 57, sign: false });
    data.append(i32 { mag: 58, sign: false });
    data.append(i32 { mag: 59, sign: false });
    data.append(i32 { mag: 60, sign: false });
    data.append(i32 { mag: 61, sign: false });
    data.append(i32 { mag: 62, sign: false });
    data.append(i32 { mag: 63, sign: false });
    data.append(i32 { mag: 64, sign: false });
    data.append(i32 { mag: 65, sign: false });
    data.append(i32 { mag: 66, sign: false });
    data.append(i32 { mag: 67, sign: false });
    data.append(i32 { mag: 68, sign: false });
    data.append(i32 { mag: 69, sign: false });
    data.append(i32 { mag: 70, sign: false });
    data.append(i32 { mag: 71, sign: false });
    data.append(i32 { mag: 72, sign: false });
    data.append(i32 { mag: 73, sign: false });
    data.append(i32 { mag: 74, sign: false });
    data.append(i32 { mag: 75, sign: false });
    data.append(i32 { mag: 76, sign: false });
    data.append(i32 { mag: 77, sign: false });
    data.append(i32 { mag: 78, sign: false });
    data.append(i32 { mag: 79, sign: false });
    data.append(i32 { mag: 80, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}