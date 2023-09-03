use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;use orion::numbers::FP8x23;


fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 276824064, sign: false });
    data.append(FP8x23 { mag: 889192448, sign: false });
    data.append(FP8x23 { mag: 117440512, sign: false });
    data.append(FP8x23 { mag: 402653184, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}