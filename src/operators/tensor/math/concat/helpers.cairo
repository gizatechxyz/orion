use core::clone::Clone;
use array::{ArrayTrait, SpanTrait};
use option::OptionTrait;
use debug::PrintTrait;
use core::traits::Into;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::implementations::{impl_tensor_i32, impl_tensor_u32};
use orion::operators::tensor::helpers::replace_index;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

/// Cf: TensorTrait::concat_helper docstring
fn concat_helper<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TAdd: Add<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,

>(
    mut tensors: Span<Tensor<T>>, axis: usize,  
) -> Tensor<T>  {
    assert(tensors.len() >= 2, 'Input tensors must be > 1');
    let base_tensor = *tensors.at(0);

    let base_shape = base_tensor.shape;
    let dimension =  base_shape.len();
    assert(dimension > axis, 'Out of bounds for dimension');

    let tensor_len = tensors.len();
    let mut output_data = ArrayTrait::<T>::new();
    let mut output_size = ArrayTrait::<u32>::new();
    let extra = Option::<ExtraParams>::None(());

    let mut axis_index_shape: usize = 0; 
    
    let mut shape_index: usize = 0;
    loop {
        if (shape_index == tensor_len){
            break();
        };
        let current_shape_ind = 0;
        let index_shape = *tensors.at(shape_index).shape;
        assert(dimension == index_shape.len(), 'Dimension not the same');

        let mut base_index: usize = 0;
        let mut max_break: usize = 1;

        loop {
            if (base_index == dimension){
                break();
            };
            if (base_index == axis){
                assert(max_break == 1, 'More than 1 axis not same');
                axis_index_shape += *index_shape.at(base_index);
                max_break -=1;
            } else {
                assert(base_shape.at(base_index) == index_shape.at(base_index), 'Shape is not the same')
            }
            base_index +=1;
        };

        shape_index += 1;
    };

    // output size of data
    let mut index = 0;
    loop {
         if (index == dimension){
            break();
        };
        let val = *base_shape.at(index);
        if (index == axis){
            output_size.append(axis_index_shape)
        }
        else {
            output_size.append(val)
        }
        index +=1;
    };

    // Concatenation loop 
    let mut total_loop = 1;
    if (axis == 0_u32){
        total_loop == 1;
    } 
    else {
        let mut total_loop_index: usize = 0;
        loop {
            if (total_loop_index == axis){
                break();
            };
            total_loop *= *base_shape.at(total_loop_index);

            total_loop_index += 1;
        };

    }

    // Loop through dimension
    let mut outer_loop_index: usize = 0;
    loop {
        if (outer_loop_index == total_loop.into()) {
            break();
        };

        // Loop through each tensor
        let mut tensor_index: usize = 0;
        loop {
            if (tensor_index == tensor_len){
                break();
            }

            let active_tensor = *tensors.at(tensor_index);
            let total_active_tensor = active_tensor.data.len();

            let mut inner_index: usize = 0;
            let count = total_active_tensor / total_loop;
            loop {
                if (inner_index == count){
                    break();
                }

                output_data.append(*active_tensor.data.at(count*outer_loop_index + inner_index));

                inner_index += 1;
            };

            tensor_index += 1;
        };

        outer_loop_index += 1;
    };

    let mut output_tensor = TensorTrait::<T>::new(output_size.span(), output_data.span(), extra);
    
    output_tensor
}


