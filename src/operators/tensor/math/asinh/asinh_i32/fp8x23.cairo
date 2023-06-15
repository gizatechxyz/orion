use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::numbers::signed_integer::i32::i32;

use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::utils::check_gas;

/// Cf: TensorTrait::asinh docstring
fn asinh(self: @Tensor<i32>) -> Tensor<FixedType> {
    let mut result = ArrayTrait::new();
    let mut data = *self.data;

    loop {
        check_gas();

        let ele = *data.pop_front().unwrap();
        
        if ele.sign == true {
            let ele_pos = FixedTrait::from_unscaled_felt(ele.mag.into()*-1);
            let neg_ele = FixedTrait::from_unscaled_felt(ele.mag.into());

            let ele_exp = FixedTrait::exp(ele_pos);
            let neg_ele_exp = FixedTrait::exp(neg_ele);
            let sum = ele_exp + neg_ele_exp;
            let answer = sum / FixedTrait::from_unscaled_felt(2);

            result.append(answer);
        }
        else {
            let ele_pos = FixedTrait::from_unscaled_felt(ele.mag.into());
            let neg_ele = FixedTrait::from_unscaled_felt(ele.mag.into()*-1);

            let ele_exp = FixedTrait::exp(ele_pos);
            let neg_ele_exp = FixedTrait::exp(neg_ele);
            let sum = ele_exp + neg_ele_exp;
            let answer = sum / FixedTrait::from_unscaled_felt(2);

            result.append(answer);
        }

        if (data.len() == 0) {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span(), *self.extra);
}