use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: TensorTrait::array_feature_extractor docstring
fn array_feature_extractor<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut self: Tensor<T>, indices: Tensor<usize>
) -> Tensor<T> {

    assert(indices.shape.len() == 1, 'Indices must be a 1D tensor');
    
    let input_shape: Span<usize> = self.shape;
    let input_data: Span<T> = self.data;
    
    if input_shape.len() == 1 {

        let mut output_data = ArrayTrait::<T>::new();

        let mut indices_counter: usize = 0;

        loop {
            if indices_counter > indices.data.len() - 1 {
                break;
            }

            let mut current_indices_value = *indices.data.at(indices_counter);

            assert(current_indices_value < *input_shape.at(0), 'Indices out of range');

            let mut current_data_value = *input_data.at(current_indices_value);

            output_data.append(current_data_value);
            
            indices_counter += 1;
        };

        return TensorTrait::new(indices.shape, output_data.span());
    }

    let last_tensor_axis: usize = *input_shape.at(input_shape.len() - 1);

    let mut input_shape_counter: usize = 0;

    let mut total_elements: usize = 1;

    let mut output_shape: Array<usize> = ArrayTrait::new();

    loop {
        if input_shape_counter > input_shape.len() - 2 {
            break;
        }

        let mut current_shape_value = *input_shape.at(input_shape_counter);

        output_shape.append(current_shape_value);

        total_elements = total_elements * current_shape_value;

        input_shape_counter += 1;

    };

    output_shape.append(indices.data.len());

    let mut output_data = ArrayTrait::<T>::new();

    let strides: Span<usize> = TensorTrait::stride(@self);

    let mut element_counter: usize = 0;

    loop {
        if element_counter > total_elements - 1 {
            break;
        }

        let mut base_index = if strides.len() > 1 {
            element_counter * (*strides.at(strides.len() - 2))
        } else {
            0  
        };

        let mut indices_counter: usize = 0;

        loop {
            if indices_counter > indices.data.len() - 1 {
                break;
            }

            let mut current_indices_value = *indices.data.at(indices_counter);

            assert(current_indices_value < last_tensor_axis, 'Indices out of range');

            let mut flat_index = base_index + current_indices_value * (*strides.at(strides.len() - 1));

            let mut current_data_value = *input_data.at(flat_index);

            output_data.append(current_data_value);
            
            indices_counter += 1;

        };

        element_counter += 1;
    };

    return TensorTrait::new(output_shape.span(), output_data.span());

}