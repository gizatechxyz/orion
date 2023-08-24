use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16Impl, FP16x16Add, FP16x16Mul, FP16x16Div
};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};


/// Cf: NNTrait::sigmoid docstring
fn sigmoid_fp(mut z: Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let result = FixedTrait::ONE()
                    / (FixedTrait::ONE() + (*item * FixedType { mag: 65536, sign: true }).exp());
                data_result.append(result);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<FixedType>::new(z.shape, data_result.span(), z.extra);
}

