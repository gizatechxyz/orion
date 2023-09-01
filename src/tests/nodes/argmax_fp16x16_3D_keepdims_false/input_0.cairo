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
    data.append(FP16x16 { mag: 4128768, sign: false });
    data.append(FP16x16 { mag: 4390912, sign: true });
    data.append(FP16x16 { mag: 7274496, sign: true });
    data.append(FP16x16 { mag: 5111808, sign: true });
    data.append(FP16x16 { mag: 6684672, sign: true });
    data.append(FP16x16 { mag: 2818048, sign: true });
    data.append(FP16x16 { mag: 6553600, sign: true });
    data.append(FP16x16 { mag: 3407872, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}