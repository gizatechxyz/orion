use core::option::OptionTrait;
use core::traits::TryInto;
use core::traits::Into;

use array::ArrayTrait;
use array::SpanTrait;

use orion::numbers::signed_integer::integer_trait::IntegerTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices, get_all_axes};

use alexandria_sorting::bubble_sort;
use alexandria_data_structures::array_ext::{SpanTraitExt};


/// Cf: TensorTrait::reduce_min docstring
fn reduce_min<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>,
    axes: Option<Span<usize>>,
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>
) -> Tensor<T> {
    let noop_with_empty_axes = match noop_with_empty_axes {
        Option::Some(noop_with_empty_axes) => noop_with_empty_axes,
        Option::None(_) => {
            false
        },
    };
    let axes = match axes {
        Option::Some(axes) => {
            if(axes.len() == 0) {
                get_all_axes(*self.shape)
            }
            else {
                assert(axes.len() == axes.unique().len(), 'duplicated axis.');
                let mut axes_arr = ArrayTrait::new();
                let mut copy_axes = axes;
                loop {
                    match copy_axes.pop_front() {
                        Option::Some(axis) => {
                            axes_arr.append(*axis);
                        },
                        Option::None(_) => {
                            break;
                        }
                    };
                };
                let sorted_axes = bubble_sort::bubble_sort_elements(axes_arr).span();
                sorted_axes
            }
        },
        Option::None(_) => {
            if (noop_with_empty_axes == true) {
                return *self;
            }
            get_all_axes(*self.shape)
        },
    };
    let keepdims = match keepdims {
        Option::Some(keepdims) => keepdims,
        Option::None(_) => {
            true
        },
    };
    
    let mut axis_c = 0;
    let mut copy_axes = axes;
    let mut shape = *self.shape;
    let mut data = *self.data;
    loop {
        match copy_axes.pop_front() {
            Option::Some(axis) => {
                if (shape.len() == 1) {
                    let current_min = accumulate_min::<T>(data, shape, shape, 0);
                    shape = array![].span();
                    data = array![current_min].span();
                    break();
                }
                let mut temp_data = ArrayTrait::new();
                let mut temp_shape = reduce_output_shape(shape, *axis-axis_c, false);
                let data_len = len_from_shape(temp_shape);
                let mut index: usize = 0;
                loop {
                    let indices = unravel_index(index, temp_shape);
                    let current_min = accumulate_min::<T>(data, shape, indices, *axis-axis_c);

                    temp_data.append(current_min);

                    index += 1;
                    if index == data_len {
                        break ();
                    };
                };
                shape = temp_shape;
                data = temp_data.span();
                axis_c += 1;
            },
            Option::None(_) => {
                break;
            }
        };
    };
    
    let mut axes_copy = axes;
    if keepdims == true {
        shape = *self.shape;
        loop {
            match axes_copy.pop_front() {
                Option::Some(axis) => {
                    shape = reduce_output_shape(shape, *axis, true);
                },
                Option::None(_) => {
                    break;
                }
            };
        };
        return TensorTrait::<T>::new(shape, data);
    } else {
        return TensorTrait::<T>::new(shape, data);
    }
}

/// Helper function that accumulates the minimum of elements along a specific axis.
///
/// # Arguments
/// * `input_data` - The input's data.
/// * `input_shape` - The input's shape.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to accumulate the minimum.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A value representing the accumulated minimum along the specified axis.
fn accumulate_min<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut input_data: Span<T>, input_shape: Span<usize>, output_indices: Span<usize>, axis: usize
) -> T {
    let axis_len = *(input_shape)[axis];
    let mut min: T = NumberTrait::max_value();

    let mut axis_index = 0;

    if (input_shape).len() > 1 {
        loop {
            if axis_index == axis_len {
                break ();
            }

            let input_indices = combine_indices(output_indices, axis_index, axis);
            let input_index = ravel_index(input_shape, input_indices);
            let ele = *(input_data)[input_index];
            if (ele < min) {
                min = ele;
            }

            axis_index += 1;
        };
    } else {
        loop {
            match input_data.pop_front() {
                Option::Some(item) => { 
                    if (*item < min) {
                        min = *item;
                    }
                },
                Option::None(_) => { break; }
            };
        };
    }
    return min;
}
