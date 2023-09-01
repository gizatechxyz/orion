use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;use orion::numbers::FP8x23;


fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 92274688, sign: false });
    data.append(FP8x23 { mag: 218103808, sign: false });
    data.append(FP8x23 { mag: 545259520, sign: true });
    data.append(FP8x23 { mag: 855638016, sign: false });
    data.append(FP8x23 { mag: 377487360, sign: true });
    data.append(FP8x23 { mag: 209715200, sign: true });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 654311424, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}