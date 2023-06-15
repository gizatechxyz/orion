use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::numbers::signed_integer::i32::i32;

use orion::numbers::fixed_point::implementations::impl_16x16;
use orion::utils::check_gas;

/// Cf: TensorTrait::asinh docstring
fn asinh(self: @Tensor<u32>) -> Tensor<FixedType> {
    let mut result = ArrayTrait::new();
    let mut data = *self.data;

    loop {
        check_gas();

        let ele = *data.pop_front().unwrap();
        
        let x = FixedTrait::from_unscaled_felt(ele.into());
        let x_sq_plus_one = (x * x) + FixedTrait::from_unscaled_felt(1);
        let sqrt = x_sq_plus_one.sqrt();
        let sqrt_plus_x = sqrt + x;
        let answer = sqrt_plus_x.ln();
        result.append(answer);

        if (data.len() == 0) {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span(), *self.extra);
}