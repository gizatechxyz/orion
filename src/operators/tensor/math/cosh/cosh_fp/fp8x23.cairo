use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;

use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::utils::check_gas;

/// Cf: TensorTrait::cosh docstring
fn cosh(self: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut result = ArrayTrait::new();
    let mut data = *self.data;

    loop {
        check_gas();

        let ele = *data.pop_front().unwrap();
        let neg_ele = ele * FixedTrait::new(1, true);
        let ele_exp = FixedTrait::exp(ele);
        let neg_ele_exp = FixedTrait::exp(neg_ele);
        let sum = ele_exp + neg_ele_exp;

        result.append(sum / FixedTrait::new(2, false));

        if (data.len() == 0) {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span(), *self.extra);
}
