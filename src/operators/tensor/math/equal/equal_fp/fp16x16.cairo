use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16PartialEq;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

use orion::operators::tensor::core::{Tensor, TensorTrait};

use orion::operators::tensor::helpers::check_compatibility;


/// Cf: TensorTrait::equal docstring
fn equal(y: @Tensor<FixedType>, z: @Tensor<FixedType>) -> Tensor<usize> {
    check_compatibility(*y.shape, *z.shape);

    let mut data_result = ArrayTrait::<usize>::new();
    let (mut smaller, mut bigger) = if (*y.data).len() < (*z.data).len() {
        (y, z)
    } else {
        (z, y)
    };

    let mut bigger_data = *bigger.data;
    let mut smaller_data = *smaller.data;
    let mut smaller_index = 0;

    loop {
        if bigger_data.len() == 0 {
            break ();
        };

        let bigger_current_index = *bigger_data.pop_front().unwrap();
        let smaller_current_index = *smaller_data[smaller_index];

        if bigger_current_index == smaller_current_index {
            data_result.append(1);
        } else {
            data_result.append(0);
        };

        smaller_index = (1 + smaller_index) % smaller_data.len();
    };

    return TensorTrait::<usize>::new(*bigger.shape, data_result.span(), *y.extra);
}
