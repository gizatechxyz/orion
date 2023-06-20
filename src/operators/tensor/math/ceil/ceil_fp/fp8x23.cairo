use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::utils::check_gas;


/// Cf: TensorTrait::ceil docstring
fn ceil(z: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    loop {
        check_gas();

        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        data_result.append(current_index.ceil());
    };

    return TensorTrait::new(*z.shape, data_result.span(), *z.extra);
}
