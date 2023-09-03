use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;use orion::numbers::FP8x23;


fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 780140544, sign: true });
    data.append(FP8x23 { mag: 897581056, sign: false });
    data.append(FP8x23 { mag: 201326592, sign: true });
    data.append(FP8x23 { mag: 662700032, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}