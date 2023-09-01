use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;
use orion::numbers::FP16x16;


fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 406504, sign: false });
    data.append(FP16x16 { mag: 169826, sign: true });
    data.append(FP16x16 { mag: 286936, sign: false });
    data.append(FP16x16 { mag: 216587, sign: true });
    data.append(FP16x16 { mag: 320888, sign: true });
    data.append(FP16x16 { mag: 186065, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}