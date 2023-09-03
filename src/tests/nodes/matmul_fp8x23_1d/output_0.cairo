use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;use orion::numbers::FP8x23;


fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16777216, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}