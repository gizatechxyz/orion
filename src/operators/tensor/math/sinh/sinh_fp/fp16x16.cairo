use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;

use orion::numbers::fixed_point::implementations::impl_16x16;
use orion::utils::check_gas;

/// Cf: TensorTrait::sinh docstring
fn sinh(self: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut result = ArrayTrait::<FixedType>::new();
    let mut data = *self.data;

    loop {
        check_gas();

        let ele = *data.pop_front().unwrap();

        if ele.sign == true {
            let ele_pos = FixedTrait::new(ele.mag, true); 
            let neg_ele = ele_pos * FixedTrait::new_unscaled(1, true);
            let ele_exp = FixedTrait::exp(ele_pos);
            let neg_ele_exp = FixedTrait::exp(neg_ele);
            let diff = ele_exp - neg_ele_exp;
            let answer = diff / FixedTrait::new_unscaled(2, false);
            result.append(answer);
        }
        else {        
            let ele_pos = FixedTrait::new(ele.mag, false); 
            let neg_ele = ele_pos * FixedTrait::new_unscaled(1, true);
            let ele_exp = FixedTrait::exp(ele_pos);
            let neg_ele_exp = FixedTrait::exp(neg_ele);
            let diff = ele_exp - neg_ele_exp;
            let answer = diff / FixedTrait::new_unscaled(2, false);
            result.append(answer);
        }
        if (data.len() == 0) {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span(), *self.extra);
}
