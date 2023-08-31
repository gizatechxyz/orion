use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;
use orion::numbers::FP16x16;


fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6422528, sign: true });
    data.append(FP16x16 { mag: 5373952, sign: true });
    data.append(FP16x16 { mag: 3473408, sign: true });
    data.append(FP16x16 { mag: 5505024, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}