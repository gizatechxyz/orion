use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;
use orion::numbers::FP16x16;


fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6553600, sign: false });
    data.append(FP16x16 { mag: 7667712, sign: true });
    data.append(FP16x16 { mag: 6815744, sign: true });
    data.append(FP16x16 { mag: 6488064, sign: true });
    data.append(FP16x16 { mag: 7274496, sign: false });
    data.append(FP16x16 { mag: 6750208, sign: false });
    data.append(FP16x16 { mag: 7405568, sign: false });
    data.append(FP16x16 { mag: 720896, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}