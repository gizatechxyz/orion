use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
use orion::numbers::FP16x16;


fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 62427, sign: false });
    data.append(FP16x16 { mag: 3108, sign: false });
    data.append(FP16x16 { mag: 57723, sign: false });
    data.append(FP16x16 { mag: 7812, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}