use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
use orion::numbers::FP16x16;


fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7143424, sign: false });
    data.append(FP16x16 { mag: 7471104, sign: true });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 1835008, sign: false });
    data.append(FP16x16 { mag: 2883584, sign: false });
    data.append(FP16x16 { mag: 3014656, sign: true });
    data.append(FP16x16 { mag: 3145728, sign: true });
    data.append(FP16x16 { mag: 4521984, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}