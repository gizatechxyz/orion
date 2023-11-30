use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: TensorTrait::array_feature_extractor docstring
fn array_feature_extractor<
    T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>
>(
    self: Tensor<T>, indices: Tensor<usize>
) -> Tensor<T> {
    assert(indices.shape.len() == 1, 'Indices must be a 1D tensor');

    if self.shape.len() == 1 {
        return process_1D_tensor(self, indices);
    }

    let (output_shape, total_elements) = calculate_output_shape::<T>(self.shape, indices);

    let output_data = calculate_output_data::<T>(self, indices, total_elements);

    return TensorTrait::new(output_shape.span(), output_data.span());
}


fn process_1D_tensor<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    self: Tensor<T>, indices: Tensor<usize>
) -> Tensor<T> {
    let mut output_data = ArrayTrait::<T>::new();

    let mut indices_counter: usize = 0;

    let mut indices_values: Span<usize> = indices.data;
    let self_len = *self.shape.at(0);
    loop {
        match indices_values.pop_front() {
            Option::Some(current_indices_value) => {
                assert(*current_indices_value < self_len, 'Indices out of range');
                let mut current_data_value = *self.data.at(*current_indices_value);
                output_data.append(current_data_value);
            },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(indices.shape, output_data.span());
}


fn calculate_output_shape<
    T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>
>(
    input_shape: Span<usize>, indices: Tensor<usize>
) -> (Array<usize>, usize) {
    let mut total_elements: usize = 1;
    let mut output_shape: Array<usize> = ArrayTrait::new();

    let mut input_shape_copy = input_shape;
    let mut input_shape_counter: usize = 0;
    let breaker = input_shape.len() - 2;
    loop {
        match input_shape_copy.pop_front() {
            Option::Some(current_shape_value) => {
                if input_shape_counter > breaker {
                    break;
                }
                output_shape.append(*current_shape_value);
                total_elements = total_elements * *current_shape_value;

                input_shape_counter += 1;
            },
            Option::None(_) => { break; }
        };
    };

    output_shape.append(indices.data.len());

    return (output_shape, total_elements);
}


fn calculate_output_data<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    self: Tensor<T>, indices: Tensor<usize>, total_elements: usize
) -> Array<T> {
    let last_tensor_axis: usize = *self.shape.at(self.shape.len() - 1);

    let mut output_data = ArrayTrait::<T>::new();

    let strides: Span<usize> = TensorTrait::stride(@self);

    let mut element_counter: usize = 0;
    let mut stride_l2 = *strides.at(strides.len() - 2);
    let mut stride_l1 = *strides.at(strides.len() - 1);
    loop {
        if element_counter > total_elements - 1 {
            break;
        }

        let mut base_index = if strides.len() > 1 {
            element_counter * stride_l2
        } else {
            0
        };

        let mut indices_values = indices.data;
        loop {
            match indices_values.pop_front() {
                Option::Some(current_indices_value) => {
                    assert(*current_indices_value < last_tensor_axis, 'Indices out of range');
                    let mut flat_index = base_index + *current_indices_value * (stride_l1);

                    let mut current_data_value = *self.data.at(flat_index);
                    output_data.append(current_data_value);
                },
                Option::None(_) => { break; }
            };
        };

        element_counter += 1;
    };

    return output_data;
}
