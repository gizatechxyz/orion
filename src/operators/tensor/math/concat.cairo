use core::clone::Clone;
use array::{ArrayTrait, SpanTrait};
use option::OptionTrait;
use debug::PrintTrait;
use core::traits::Into;

use orion::operators::tensor::helpers::replace_index;
use orion::operators::tensor::{TensorTrait, Tensor};


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
