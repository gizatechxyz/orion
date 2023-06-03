use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::utils::check_gas;
use orion::operators::tensor::helpers::check_compatibility;


/// Cf: TensorTrait::greater docstring
fn greater(y: @Tensor<FixedType<fp8x23>>, z: @Tensor<FixedType<fp8x23>>) -> Tensor<usize> {

    check_compatibility(*y.shape,*z.shape);

    let mut data_result = ArrayTrait::<usize>::new();
    let (mut smaller, mut bigger, retains_input_order) = if (*y.data).len() < (*z.data).len() { 
        (y, z, true) 
    } else { 
        (z, y, false)
    };

    let mut bigger_data = *bigger.data;
    let mut smaller_data = *smaller.data;
    let mut smaller_index = 0;
 
    loop {
        check_gas();

        if bigger_data.len() == 0 {
            break ();
        };

        let bigger_current_index = *bigger_data.pop_front().unwrap();
        let smaller_current_index = *smaller_data.at(smaller_index);

        let (y_value, z_value) = if retains_input_order {
            (smaller_current_index, bigger_current_index)
        } else {
            (bigger_current_index, smaller_current_index)
        };
        
        if y_value > z_value {
            data_result.append(1);
        } else {
            data_result.append(0);
        };
        
        smaller_index = (1 + smaller_index) % smaller_data.len() ;
    };

    return TensorTrait::<usize>::new(*bigger.shape, data_result.span());
}