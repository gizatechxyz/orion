use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;
use orion::numbers::FP16x16;


fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 20529, sign: true });
    data.append(FP16x16 { mag: 86065, sign: true });
    data.append(FP16x16 { mag: 139390, sign: true });
    data.append(FP16x16 { mag: 8318, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}