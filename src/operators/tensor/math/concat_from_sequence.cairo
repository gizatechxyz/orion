use core::clone::Clone;
use array::{ArrayTrait, SpanTrait};
use option::OptionTrait;
use debug::PrintTrait;
use core::traits::Into;

use orion::operators::tensor::helpers::replace_index;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::signed_integer::i32::i32;


fn concat_from_sequence<T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,>(
    sequence: Array<Tensor<T>>, axis: i32, new_axis: Option<usize>
) -> Tensor<T> {

    let new_axis: usize = match new_axis {
        Option::Some(p) => {
            assert(p == 0 || p == 1, 'new_axis must be 0 or 1');
            p
        },
        Option::None(_) => 0
    };

    let first_tensor = *sequence.at(0);
    let r = first_tensor.shape.len();

    let axis_is_negative: bool = axis.sign;
    let mut axis_value: u32 = axis.mag;
    
    if new_axis == 0 {
        /// assert in range [-r, r - 1]
        assert((axis_is_negative == false && axis_value <= r - 1) || (axis_is_negative == true && axis_value <= r), 'Out of bounds for dimension');
        if axis_is_negative == true {
            axis_value = r - axis_value
        }
        concat(sequence.span(), axis_value)
    } else {
        /// assert in range [-r - 1, r]
        assert((axis_is_negative == false && axis_value <= r) || (axis_is_negative == true && axis_value <= r + 1), 'Out of bounds for dimension');
        if axis_is_negative == true {
            if axis_value > r {
                axis_value = 0
            } else {
                axis_value = r - axis_value
            }
        }
        let mut input_sequence_copy = sequence;
        let mut reshaped_sequence = ArrayTrait::<Tensor<T>>::new();
        loop {
            match input_sequence_copy.pop_front() {
                Option::Some(input_sequence_value) => {
                    let mut reshaped_tensor = add_new_dimension(input_sequence_value, axis_value);
                    reshaped_sequence.append(reshaped_tensor);
                },
                Option::None(_) => { break; }
            };
        };
        concat(reshaped_sequence.span(), axis_value)
    }
}


fn add_new_dimension<T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,>(
    mut tensor: Tensor<T>, axis: usize
) -> Tensor<T> {
    let mut tensor_shape = tensor.shape;
    let mut new_tensor_shape = ArrayTrait::<usize>::new();
    let mut tensor_shape_counter: usize = 0;
    loop {
        match tensor_shape.pop_front() {
            Option::Some(tensor_shape_value) => {
                if tensor_shape_counter == axis {
                    new_tensor_shape.append(1);
                } 
                new_tensor_shape.append(*tensor_shape_value);
                tensor_shape_counter += 1;
            },
            Option::None(_) => { break; }
        };
    };
    if axis >= tensor.shape.len() {
        new_tensor_shape.append(1);
    }
    TensorTrait::<T>::new(new_tensor_shape.span(), tensor.data)
}


fn concat<T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,>(
    mut tensors: Span<Tensor<T>>, axis: usize
) -> Tensor<T> {
    assert(tensors.len() >= 2, 'Input tensors must be > 1');
    let base_tensor = *tensors.at(0);
    let base_shape = base_tensor.shape;
    let dimension = base_shape.len();
    assert(dimension > axis, 'Out of bounds for dimension');

    // Validate shapes of tensors
    validate_shapes(tensors, base_shape, axis);

    // Calculate output size
    let output_size = compute_output_size(base_shape, tensors, axis);

    // Concatenate tensor data
    let output_data: Array<T> = concatenate_data(tensors, axis, base_shape);

    TensorTrait::<T>::new(output_size.span(), output_data.span())
}

fn validate_shapes<T>(mut tensors: Span<Tensor<T>>, mut base_shape: Span<usize>, axis: usize) {
    loop {
        match tensors.pop_front() {
            Option::Some(tensor) => {
                assert(base_shape.len() == (*tensor.shape).len(), 'Dimension not the same');

                let mut axis_index = 0;
                let mut tensor_shape = *tensor.shape;
                let mut base_shape_copy = base_shape;
                loop {
                    match tensor_shape.pop_front() {
                        Option::Some(tensor_shape_i) => {
                            let base_shape_i = base_shape_copy.pop_front().unwrap();
                            if axis_index != axis {
                                assert(base_shape_i == tensor_shape_i, 'Shape is not the same');
                            }
                            axis_index += 1;
                        },
                        Option::None(_) => { break; }
                    };
                };
            },
            Option::None(_) => { break; }
        };
    };
}

fn compute_output_size<T>(
    mut base_shape: Span<usize>, mut tensors: Span<Tensor<T>>, axis: usize
) -> Array<u32> {
    let mut output_size = ArrayTrait::<u32>::new();

    let mut axis_size = 0;
    loop {
        match tensors.pop_front() {
            Option::Some(tensor) => { axis_size += *(*tensor.shape).at(axis); },
            Option::None(_) => { break; }
        };
    };

    let mut shape_index = 0;
    loop {
        match base_shape.pop_front() {
            Option::Some(item) => {
                if shape_index == axis {
                    output_size.append(axis_size);
                } else {
                    output_size.append(*item);
                }
                shape_index += 1;
            },
            Option::None(_) => { break; }
        };
    };

    output_size
}

fn concatenate_data<T, impl TCopy: Copy<T>, impl TDrop: Drop<T>,>(
    mut tensors: Span<Tensor<T>>, axis: usize, base_shape: Span<usize>
) -> Array<T> {
    let mut output_data = ArrayTrait::<T>::new();

    let total_loops = product_upto(base_shape, axis);

    let mut outer_loop_index = 0;
    loop {
        if outer_loop_index == total_loops {
            break;
        }

        let mut tensors_copy = tensors;
        loop {
            match tensors_copy.pop_front() {
                Option::Some(tensor) => {
                    let slice_len = (*tensor.data).len() / total_loops;

                    let mut inner_index = 0;
                    loop {
                        if inner_index == slice_len {
                            break;
                        }

                        output_data
                            .append(*(*tensor.data).at(slice_len * outer_loop_index + inner_index));
                        inner_index += 1;
                    };
                },
                Option::None(_) => { break; }
            };
        };

        outer_loop_index += 1;
    };

    output_data
}

fn product_upto(mut shape: Span<usize>, upto: usize) -> usize {
    let mut total = 1;
    let mut index = 0;

    loop {
        match shape.pop_front() {
            Option::Some(val) => {
                if index == upto {
                    break;
                }

                total *= *val;
                index += 1;
            },
            Option::None(_) => { break; }
        };
    };

    total
}
