use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;use orion::numbers::FP8x23;


fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 134217728, sign: true });
    data.append(FP8x23 { mag: 511705088, sign: true });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 167772160, sign: true });
    data.append(FP8x23 { mag: 746586112, sign: false });
    data.append(FP8x23 { mag: 293601280, sign: true });
    data.append(FP8x23 { mag: 461373440, sign: true });
    data.append(FP8x23 { mag: 402653184, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}