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
    data.append(FP16x16 { mag: 6553600, sign: true });
    data.append(FP16x16 { mag: 5570560, sign: false });
    data.append(FP16x16 { mag: 2097152, sign: true });
    data.append(FP16x16 { mag: 7602176, sign: true });
    data.append(FP16x16 { mag: 5373952, sign: false });
    data.append(FP16x16 { mag: 1179648, sign: true });
    data.append(FP16x16 { mag: 2162688, sign: true });
    data.append(FP16x16 { mag: 3080192, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}