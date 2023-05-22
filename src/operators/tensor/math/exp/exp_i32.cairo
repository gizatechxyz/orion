use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::fixed_point::types::{Fixed, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::utils::check_gas;

/// Cf: TensorTrait::exp docstring
fn exp(self: @Tensor<i32>) -> Tensor<FixedType> {
    let mut result = ArrayTrait::new();
    let mut data = *self.data;

    loop {
        check_gas();

        let ele = *data.pop_front().unwrap();

        if ele.sign == true {
            let ele = Fixed::from_unscaled_felt((ele.mag).into() * -1);
            result.append(Fixed::exp(ele))
        } else {
            let ele = Fixed::from_unscaled_felt((ele.mag).into());
            result.append(Fixed::exp(ele))
        }

        if (data.len() == 0) {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span());
}
