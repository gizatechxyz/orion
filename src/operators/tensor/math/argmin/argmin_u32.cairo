use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::argmin::helpers::{find_argmin_1D,find_argmin};
use orion::utils::check_gas;

/// Cf: TensorTrait::argmin docstring
fn argmin(
    self: @Tensor<u32>, 
    axis: usize, 
    keepdims: Option<bool>, 
    select_last_index:Option<bool> 
    ) -> Tensor<usize> {

    let keepdims = match keepdims {
        Option::Some(val) => val,
        Option::None(_) => true,
    };

    let select_last_index = match select_last_index {
        Option::Some(val) => val,
        Option::None(_) => false,
    };
    
    assert(axis <= (*self.shape).len(), 'axis out of dimensions');

    if (*self.shape).len() == 1 { 
        return find_argmin_1D(self, axis, true, select_last_index);
    }

    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_output_shape(*self.shape, axis, false);
    let output_data_len = len_from_shape(output_shape);

    let MAX = 4294967295_u32; // 2 ** 32 - 1

    let mut index: usize = 0;
    loop {
        check_gas();

        let output_indices = unravel_index(index, output_shape);
        let current_argmin = find_argmin(
            self, output_indices, axis, 0, MAX, 0, select_last_index
        );

        output_data.append(current_argmin);

        index += 1;
        if index == output_data_len {
            break ();
        };
    };

    return TensorTrait::<usize>::new(reduce_output_shape(*self.shape, axis, keepdims), output_data.span(), *self.extra);
}
