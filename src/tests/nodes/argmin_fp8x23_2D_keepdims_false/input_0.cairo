use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;use orion::numbers::FP8x23;


fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 595591168, sign: true });
    data.append(FP8x23 { mag: 922746880, sign: true });
    data.append(FP8x23 { mag: 75497472, sign: true });
    data.append(FP8x23 { mag: 360710144, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}