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
    data.append(FP16x16 { mag: 4653056, sign: false });
    data.append(FP16x16 { mag: 6750208, sign: true });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 6160384, sign: true });
    data.append(FP16x16 { mag: 4063232, sign: true });
    data.append(FP16x16 { mag: 3932160, sign: true });
    data.append(FP16x16 { mag: 8257536, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}