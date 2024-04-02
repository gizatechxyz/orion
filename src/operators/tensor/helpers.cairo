use alexandria_data_structures::array_ext::ArrayTraitExt;

use orion::utils::u32_max;
use orion::operators::tensor::{core::{Tensor, TensorTrait, stride}, BoolTensor};

/// Calculates the number of elements in a tensor given its shape.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize representing the number of elements in the tensor.
fn len_from_shape(mut shape: Span<usize>) -> usize {
    let mut result: usize = 1;

    loop {
        match shape.pop_front() {
            Option::Some(item) => { result *= *item; },
            Option::None => { break; }
        };
    };

    result
}

/// Verifies if the shape and the data array of a tensor are compatible.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
/// * `data` -  A span containing the data elements of generic type T.
///
/// # Panics
/// * Panics if the shape and data array are incompatible.
fn check_shape<T>(shape: Span<usize>, data: Span<T>) {
    assert(len_from_shape(shape) == data.len(), 'wrong tensor shape');
}

/// Checks if two tensor shapes are compatible for broadcasting.
///
/// # Arguments
/// * `shape_1` - A span containing the first tensor's shape as usize elements.
/// * `shape_2` - A span containing the second tensor's shape as usize elements.
///
/// # Panics
/// * Panics if the shapes are not compatible for broadcasting.
fn check_compatibility(mut shape_1: Span<usize>, mut shape_2: Span<usize>) {
    // Start from the last dimension by getting the length of each shape
    let mut iter_1 = shape_1.len();
    let mut iter_2 = shape_2.len();

    // Iterate while there are dimensions left in either shape
    while iter_1 > 0
        || iter_2 > 0 {
            // Get the current dimension for each shape, defaulting to 1 if we've run out of dimensions
            let dim_1 = if iter_1 > 0 {
                *shape_1[iter_1 - 1]
            } else {
                1
            };
            let dim_2 = if iter_2 > 0 {
                *shape_2[iter_2 - 1]
            } else {
                1
            };

            // Check the broadcasting rule for the current dimension
            if dim_1 != dim_2 && dim_1 != 1 && dim_2 != 1 {
                panic(array!['tensors shape must match']);
            }

            // Move to the next dimension
            if iter_1 > 0 {
                iter_1 -= 1;
            }
            if iter_2 > 0 {
                iter_2 -= 1;
            }
        }
}

/// Computes the index in the broadcasted tensor corresponding to the given indices and shape.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
/// * `indices` - A span containing the indices as usize elements.
///
/// # Panics
/// * Panics if shape and indices length are not equal.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize representing the index in the broadcasted tensor.
fn broadcast_index_mapping(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    if shape.len() == indices.len() {
        broadcast_index_mapping_equal_shape(shape, indices)
    } else {
        broadcast_index_mapping_non_equal_shape(shape, indices)
    }
}


fn broadcast_index_mapping_equal_shape(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    let mut result = 0_usize;
    let mut stride = stride(shape);

    loop {
        match shape.pop_front() {
            Option::Some(shape_val) => {
                let indices_val = *indices.pop_front().unwrap();
                let stride_val = *stride.pop_front().unwrap();

                let index = (indices_val % *shape_val) * stride_val;
                result += index;
            },
            Option::None => { break; }
        };
    };

    result
}

fn broadcast_index_mapping_non_equal_shape(
    mut shape: Span<usize>, mut indices: Span<usize>
) -> usize {
    let mut result = 0_usize;
    let mut stride = stride(shape.clone());

    // Calculate the offset to align indices with the rightmost dimensions of the shape
    let mut offset = if shape.len() > indices.len() {
        shape.len() - indices.len()
    } else {
        0
    };

    loop {
        match shape.pop_back() {
            Option::Some(_) => {
                let stride_val = stride
                    .pop_back()
                    .unwrap_or(@1); // Default stride for non-existent dimensions is 1

                // Calculate the index, using 0 for dimensions beyond the length of indices
                let index_val = if offset > 0 {
                    offset -= 1; // Decrement offset until we align indices with the shape
                    0 // Use 0 for indices beyond the length of the indices span
                } else {
                    *indices
                        .pop_back()
                        .unwrap_or(@0) // Use actual index value or 0 if indices are exhausted
                };

                let index = index_val * *stride_val;
                result += index;
            },
            Option::None => { break; }
        };
    };

    result
}


/// Generates the output shape after reducing a tensor along a specified axis.
///
/// # Arguments
/// * `input_shape` - A span containing the input tensor's shape as usize elements.
/// * `axis` - A usize representing the axis to reduce.
///
/// # Panics
/// * Panics if input_shape is empty.
/// * Panic if the axis is not in the valid range of the input_shape dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize representing the output shape after reduction.
fn reduce_output_shape(mut input_shape: Span<usize>, axis: usize, keepdims: bool) -> Span<usize> {
    assert(axis < input_shape.len(), 'axis out of dimensions');

    let mut output_shape: Array<u32> = array![];
    let mut n: usize = 0;

    loop {
        match input_shape.pop_front() {
            Option::Some(current_dim) => {
                if n == axis {
                    if keepdims {
                        output_shape.append(1);
                    }
                } else {
                    output_shape.append(*current_dim);
                }

                n += 1;
            },
            Option::None => { break; }
        };
    };

    output_shape.span()
}


/// Helper function that computes the output shape of a tensor after applying the axes permutation.
///
/// # Arguments
/// * `input_shape` - A span containing the input tensor's shape as usize elements.
/// * `axes` -  A span containing the usize elements representing the axes permutation.
///
/// # Panics
/// * Panics if shape and axes length are not equal.
/// * Panic if the axis value in axes is not in the valid range of the input_shape dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize representing the output shape after permutation.
fn permutation_output_shape(input_shape: Span<usize>, mut axes: Span<usize>) -> Span<usize> {
    let axes_len = axes.len();
    assert(input_shape.len() == axes_len, 'input_shape/indices len unequal');

    let mut output_shape: Array<u32> = array![];

    loop {
        match axes.pop_front() {
            Option::Some(item) => { output_shape.append(*input_shape[*item]); },
            Option::None => { break; }
        };
    };

    output_shape.span()
}

/// Combines output indices with the current index of the specified axis.
///
/// # Arguments
/// * `output_indices` - A span containing the output indices as usize elements.
/// * `axis_index` - A usize representing the current index of the specified axis.
/// * `axis` - A usize representing the specified axis.
///
/// # Panics
/// * Panics if the axis value is not in the range of the output_indices length.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize representing the combined indices.
fn combine_indices(mut output_indices: Span<usize>, axis_index: usize, axis: usize) -> Span<usize> {
    assert(axis <= output_indices.len(), 'axis value is out of range');

    let mut result: Array<u32> = array![];
    let mut n: usize = 0;

    while n != output_indices.len()
        + 1 {
            if n == axis {
                result.append(axis_index);
            } else if n > axis {
                result.append(*output_indices[n - 1_usize]);
            } else {
                result.append(*output_indices[n]);
            }

            n += 1;
        };

    result.span()
}


/// Helper function that finds the index of a target axis in the given axes array.
///
/// # Arguments
/// * `axes` - A span containing the usize elements representing the axes.
/// * `target_axis` - A usize representing the target axis.
///
/// # Panics
/// * Panics if the target_axis value is not in the range of the axes dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize representing the index of the target axis in the given axes array.
fn find_axis(mut axes: Span<usize>, target_axis: usize) -> usize {
    assert(target_axis < axes.len(), 'target_axis is out of range');
    let mut axis: usize = 0;

    loop {
        match axes.pop_front() {
            Option::Some(item) => {
                if *item == target_axis {
                    break ();
                }
                axis += 1;
            },
            Option::None => { break; }
        };
    };

    axis
}

/// Computes the broadcasted shape of two tensors.
///
/// # Arguments
/// * `shape1` - A span containing the shape of the first tensor as usize elements.
/// * `shape2` - A span containing the shape of the second tensor as usize elements.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize representing the broadcasted shape.
fn broadcast_shape(mut shape1: Span<usize>, mut shape2: Span<usize>) -> Span<usize> {
    check_compatibility(shape1, shape2);
    let mut result: Array<usize> = array![];

    while !shape1.is_empty()
        || !shape2
            .is_empty() {
                let dim1 = *shape1.pop_back().unwrap_or(@1);
                let dim2 = *shape2.pop_back().unwrap_or(@1);

                let broadcasted_dim = u32_max(dim1, dim2);
                result.append(broadcasted_dim);
            };

    result.reverse().span()
}

/// Substitute a value in a shape at a given index
/// 
/// # Arguments
///
/// * `shape` - The shape to modify
/// * `index` - The index to modify
/// * `value` - The value to insert
///
/// # Panics
/// * Panics if the index is out of bounds
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * `Span<usize>` - The modified shape
fn replace_index(mut shape: Span<usize>, index: usize, value: usize) -> Span<usize> {
    let mut output: Array<u32> = array![];
    let mut i = 0;

    loop {
        match shape.pop_front() {
            Option::Some(item) => {
                if i == index {
                    output.append(value);
                } else {
                    output.append(*item);
                };
                i += 1;
            },
            Option::None => { break; }
        };
    };

    output.span()
}

/// Creates a list of all axes of given shape
/// 
/// # Arguments
///
/// * `shape` - A span containing the input tensor's shape as usize elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * `Span<usize>` - A span containing the usize elements representing the axes.
fn get_all_axes(shape: Span<usize>) -> Span<usize> {
    let mut ret: Array<usize> = array![];
    let mut i: usize = 0;
    let stop_i = shape.len() - 1;
    loop {
        ret.append(i);
        if i == stop_i {
            break ();
        }
        i += 1;
    };

    ret.span()
}

/// Flatten a given array of tensors into an Array<T>.
fn flatten_array_of_tensors<T, +Copy<T>, +Drop<T>,>(
    tensors: Array<Tensor<T>>, axis: usize, new_shape: Span<usize>
) -> Span<T> {
    let mut new_stride = stride(new_shape);

    let mut flattened: Array<T> = array![];

    let stride_lim: usize = *new_stride.at(axis);
    let max_row = (*(*tensors.at(0).shape).at(0));
    let mut row = 0;
    while row != max_row {
        let mut tensors_span = tensors.span();
        loop {
            let mut i = 0;
            match tensors_span.pop_front() {
                Option::Some(mut t) => {
                    let mut data = *t.data;
                    while i != stride_lim {
                        let idx = i + (row * stride_lim);
                        flattened.append(*data.at(idx));
                        i += 1;
                    }
                },
                Option::None => { break; },
            }
        };

        row += 1;
    };

    flattened.span()
}

/// Convert a Tensor to an array of tensors along a given axis.
fn as_tensors_array<T, +Copy<T>, +Drop<T>, +TensorTrait<T>,>(
    tensor: @Tensor<T>, axis: usize
) -> Array<Tensor<T>> {
    let shape = *tensor.shape;
    let rank = shape.len();
    let mut as_tensors: Array<Tensor<T>> = array![];

    let mut axes: Array<usize> = array![];
    let mut idx: usize = 0;
    while idx != rank {
        axes.append(idx);
        idx += 1;
    };

    idx = 0;
    let axis_len: usize = *shape.at(axis);
    while idx != axis_len {
        let mut starts: Array<usize> = array![];
        let mut ends: Array<usize> = array![];
        let mut i: usize = 0;
        while i != rank {
            starts.append(if i == axis {
                idx
            } else {
                0
            });
            ends.append(if i == axis {
                idx + 1
            } else {
                *shape.at(i)
            });
            i += 1;
        };

        let sub_tensor: Tensor<T> = tensor
            .slice(
                starts: starts.span(),
                ends: ends.span(),
                axes: Option::Some(axes.span()),
                steps: Option::None(())
            );

        as_tensors.append(sub_tensor);

        idx += 1;
    };

    as_tensors
}

/// Compares two Spans of generic type T.
///
/// # Returns
/// an i8 type containing:
/// * 1 if the left operand is greater than the right,
/// * 0 if the left operand is equal to the right,
/// * -1 if the left operand is lower than the right,
fn span_cmp<T, +Drop<T>, +Copy<T>, +PartialEq<T>, +PartialOrd<T>>(
    lhs: Span<T>, rhs: Span<T>
) -> i8 {
    let mut rhs = rhs;
    let mut lhs = lhs;
    let mut ret: i8 = 0;
    loop {
        match lhs.pop_front() {
            Option::Some(l) => {
                match rhs.pop_front() {
                    Option::Some(r) => { if l != r {
                        ret = if *l > *r {
                            1
                        } else {
                            -1
                        };
                        break;
                    } },
                    Option::None => {
                        ret = 1;
                        break;
                    },
                }
            },
            Option::None => {
                ret = -1;
                break;
            }
        };
    };

    ret
}

/// Implements PartiaLOrd for two spans of generic type T.
impl SpanPartialOrd<T, +Drop<T>, +Copy<T>, +PartialEq<T>, +PartialOrd<T>> of PartialOrd<Span<T>> {
    fn ge(lhs: Span<T>, rhs: Span<T>) -> bool {
        span_cmp(lhs, rhs) >= 0
    }

    fn gt(lhs: Span<T>, rhs: Span<T>) -> bool {
        span_cmp(lhs, rhs) > 0
    }

    fn le(lhs: Span<T>, rhs: Span<T>) -> bool {
        span_cmp(lhs, rhs) <= 0
    }

    fn lt(lhs: Span<T>, rhs: Span<T>) -> bool {
        span_cmp(lhs, rhs) < 0
    }
}

/// Returns true if (1) the input is an optional-type and contains an element, 
/// or, (2) the input is a tensor type.
/// If the input is not provided or is an empty optional-type, this op returns false.
///
/// # Arguments
/// * `x` - The optional input.
///
/// # Returns
/// * A scalar boolean tensor. 
/// If true, it indicates that optional-type input contains an element. Otherwise, it is empty.
fn optional_has_element<T, +Copy<T>, +Drop<T>, +TensorTrait<T>,>(
    x: Option<Tensor<T>>
) -> Tensor<bool> {
    match x {
        Option::Some => {
            let mut shape: Array<usize> = array![];
            shape.append(1);
            let mut data: Array<bool> = array![];
            data.append(true);
            TensorTrait::new(shape.span(), data.span())
        },
        Option::None => {
            let mut shape: Array<usize> = array![];
            shape.append(1);
            let mut data: Array<bool> = array![];
            data.append(false);
            TensorTrait::new(shape.span(), data.span())
        }
    }
}

/// If the input is a tensor type, it returns the input.
/// If the input is an optional type, it outputs the element in the input. 
///
/// # Arguments
/// * `x` - The optional input.
///
/// # Panics
/// * Panics if the input is an empty optional-type (i.e. does not have an element) 
///   and the behavior is undefined in this case.
///
/// # Returns
/// * Output element in the optional input.
fn optional_get_element<T, +Copy<T>, +Drop<T>, +TensorTrait<T>,>(
    x: Option<Tensor<T>>
) -> Tensor<T> {
    match x {
        Option::Some(ele) => { ele },
        Option::None => { panic(array!['The input is an empty', 'optional-type.']) }
    }
}
