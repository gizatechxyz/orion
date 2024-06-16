use core::array::{ArrayTrait, SpanTrait};
use orion_cairo::numbers::f16x16::{f16x16, FixedTrait};
use orion_cairo::tensors::Tensor;

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
pub fn len_from_shape(mut shape: Span<usize>) -> usize {
    let mut result: usize = 1;

    loop {
        match shape.pop_front() {
            Option::Some(item) => { result *= *item; },
            Option::None => { break; }
        };
    };

    result
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
pub fn broadcast_shape(mut shape1: Span<usize>, mut shape2: Span<usize>) -> Span<usize> {
    check_compatibility(shape1, shape2);
    let mut result: Array<usize> = array![];

    while !shape1.is_empty() || !shape2.is_empty() {
        let dim1 = *shape1.pop_back().unwrap_or(@1);
        let dim2 = *shape2.pop_back().unwrap_or(@1);

        let broadcasted_dim = u32_max(dim1, dim2);
        result.append(broadcasted_dim);
    };

    reverse(result)
}

fn u32_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

fn reverse(self: Array<usize>) -> Span<usize> {
    let mut data = self.span();
    let mut result = array![];

    loop {
        match data.pop_back() {
            Option::Some(item) => { result.append(*item); },
            Option::None => { break; }
        };
    };
    result.span()
}


/// Verifies if the shape and the data array of a tensor are compatible.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
/// * `data` -  A span containing the data elements of generic type T.
///
/// # Panics
/// * Panics if the shape and data array are incompatible.
pub fn check_shape<T>(shape: Span<usize>, data: Span<T>) {
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
pub fn check_compatibility(mut shape_1: Span<usize>, mut shape_2: Span<usize>) {
    // Start from the last dimension by getting the length of each shape
    let mut iter_1 = shape_1.len();
    let mut iter_2 = shape_2.len();

    // Iterate while there are dimensions left in either shape
    while iter_1 > 0 || iter_2 > 0 {
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
pub fn broadcast_index_mapping(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    if shape.len() == indices.len() {
        broadcast_index_mapping_equal_shape(shape, indices)
    } else {
        broadcast_index_mapping_non_equal_shape(shape, indices)
    }
}


pub fn broadcast_index_mapping_equal_shape(
    mut shape: Span<usize>, mut indices: Span<usize>
) -> usize {
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

pub fn broadcast_index_mapping_non_equal_shape(
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
            Option::Some(shape_val) => {
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

                let index = (index_val % *shape_val) * *stride_val;
                result += index;
            },
            Option::None => { break; }
        };
    };

    result
}


pub fn stride(mut shape: Span<usize>) -> Span<usize> {
    let mut strides = ArrayTrait::new();
    let mut stride = 1;
    loop {
        match shape.pop_back() {
            Option::Some(size) => {
                strides.append(stride);
                stride *= *size;
            },
            Option::None => { break; }
        };
    };

    reverse(strides)
}


pub fn ravel_index(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    assert(shape.len() == indices.len(), 'shape & indices length unequal');

    let mut raveled_index: usize = 0;
    let mut stride: usize = 1;

    loop {
        match shape.pop_back() {
            Option::Some(i) => {
                let index = *indices.pop_back().unwrap();
                raveled_index += index * stride;

                stride *= *i;
            },
            Option::None => { break; }
        };
    };

    raveled_index
}

pub fn unravel_index(index: usize, mut shape: Span<usize>) -> Span<usize> {
    assert(shape.len() > 0, 'shape cannot be empty');

    let mut result = ArrayTrait::new();
    let mut remainder = index;
    let mut stride = len_from_shape(shape);

    loop {
        match shape.pop_front() {
            Option::Some(i) => {
                stride /= *i;

                let coord = remainder / stride;
                remainder = remainder % stride;

                result.append(coord);
            },
            Option::None => { break; }
        };
    };

    return result.span();
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
pub fn combine_indices(
    mut output_indices: Span<usize>, axis_index: usize, axis: usize
) -> Span<usize> {
    assert(axis <= output_indices.len(), 'axis value is out of range');

    let mut result: Array<u32> = array![];
    let mut n: usize = 0;

    while n != output_indices.len() + 1 {
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
pub fn get_all_axes(shape: Span<usize>) -> Span<usize> {
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


/// Helper function that accumulates the sum of elements along a specific axis.
///
/// # Arguments
/// * `input_data` - The input's data.
/// * `input_shape` - The input's shape.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to accumulate the sum.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A value representing the accumulated sum along the specified axis.
pub fn accumulate_sum(
    mut input_data: Span<f16x16>, input_shape: Span<usize>, output_indices: Span<usize>, axis: usize
) -> f16x16 {
    let axis_len = *(input_shape)[axis];
    let mut sum = FixedTrait::ZERO();

    let mut axis_index = 0;

    if (input_shape).len() > 1 {
        while axis_index != axis_len {
            let input_indices = combine_indices(output_indices, axis_index, axis);
            let input_index = ravel_index(input_shape, input_indices);
            let ele = *(input_data)[input_index];
            sum = sum + ele;

            axis_index += 1;
        };
    } else {
        loop {
            match input_data.pop_front() {
                Option::Some(item) => sum = sum + *item,
                Option::None => { break; }
            };
        };
    }

    sum
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
pub fn reduce_output_shape(
    mut input_shape: Span<usize>, axis: usize, keepdims: bool
) -> Span<usize> {
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


// Unique
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
// 
// https://github.com/keep-starknet-strange/alexandria/blob/3041887b95cf10f9d3cd8d75326c754b331f9573/packages/data_structures/src/span_ext.cairo#L7
pub fn unique(mut self: Span<i32>) -> Array<i32> {
    let mut ret = array![];

    while let Option::Some(v) = self.pop_front() {
        if !contains(ret.span(), v) {
            ret.append(v.clone());
        }
    };

    ret
}

fn contains(mut self: Span<i32>, item: @i32) -> bool {
    loop {
        match self.pop_front() {
            Option::Some(v) => { if v == item {
                break true;
            } },
            Option::None => { break false; },
        };
    }
}

// Bubble sort
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//https://github.com/keep-starknet-strange/alexandria/blob/3041887b95cf10f9d3cd8d75326c754b331f9573/packages/sorting/src/bubble_sort.cairo#L4
pub fn bubble_sort<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(mut array: Span<T>) -> Array<T> {
    if array.len() == 0 {
        return array![];
    }
    if array.len() == 1 {
        return array![*array[0]];
    }
    let mut idx1 = 0;
    let mut idx2 = 1;
    let mut sorted_iteration = true;
    let mut sorted_array = array![];

    loop {
        if idx2 == array.len() {
            sorted_array.append(*array[idx1]);
            if sorted_iteration {
                break;
            }
            array = sorted_array.span();
            sorted_array = array![];
            idx1 = 0;
            idx2 = 1;
            sorted_iteration = true;
        } else {
            if *array[idx1] <= *array[idx2] {
                sorted_array.append(*array[idx1]);
                idx1 = idx2;
                idx2 += 1;
            } else {
                sorted_array.append(*array[idx2]);
                idx2 += 1;
                sorted_iteration = false;
            }
        };
    };
    sorted_array
}
