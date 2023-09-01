use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;use orion::numbers::FP8x23;


fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 11016437, sign: true });
    data.append(FP8x23 { mag: 17841965, sign: true });
    data.append(FP8x23 { mag: 2627829, sign: true });
    data.append(FP8x23 { mag: 1064749, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}