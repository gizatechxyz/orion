use alexandria_sorting::BubbleSort;
use alexandria_data_structures::span_ext::SpanTraitExt;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::helpers::{
    reduce_output_shape, len_from_shape, combine_indices, get_all_axes
};

/// Cf: TensorTrait::reduce_mean docstring
fn reduce_mean<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TDiv: Div<T>,
    impl TAddEq: AddEq<T>,
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
        Option::None => false,
    };
    let axes = match axes {
        Option::Some(axes) => {
            if (axes.len() == 0) {
                get_all_axes(*self.shape)
            } else {
                assert(axes.len() == axes.unique().len(), 'duplicated axis.');
                let mut axes_arr = array![];
                let mut copy_axes = axes;
                loop {
                    match copy_axes.pop_front() {
                        Option::Some(axis) => { axes_arr.append(*axis); },
                        Option::None => { break; }
                    };
                };
                let sorted_axes = BubbleSort::sort(axes_arr.span()).span();
                sorted_axes
            }
        },
        Option::None => {
            if noop_with_empty_axes {
                return *self;
            }
            get_all_axes(*self.shape)
        },
    };
    let keepdims = match keepdims {
        Option::Some(keepdims) => keepdims,
        Option::None => true,
    };

    let mut axis_c = 0;
    let mut copy_axes = axes;
    let mut shape = *self.shape;
    let mut data = *self.data;
    loop {
        match copy_axes.pop_front() {
            Option::Some(axis) => {
                if (shape.len() == 1) {
                    let current_mean = accumulate_mean::<T>(data, shape, shape, 0);
                    shape = array![].span();
                    data = array![current_mean].span();
                    break ();
                }
                let mut temp_data = array![];
                let mut temp_shape = reduce_output_shape(shape, *axis - axis_c, false);
                let data_len = len_from_shape(temp_shape);
                let mut index: usize = 0;
                while index != data_len {
                    let indices = unravel_index(index, temp_shape);
                    let current_mean = accumulate_mean::<T>(data, shape, indices, *axis - axis_c);

                    temp_data.append(current_mean);

                    index += 1;
                };

                shape = temp_shape;
                data = temp_data.span();
                axis_c += 1;
            },
            Option::None => { break; }
        };
    };

    let mut axes_copy = axes;
    if keepdims {
        shape = *self.shape;
        loop {
            match axes_copy.pop_front() {
                Option::Some(axis) => { shape = reduce_output_shape(shape, *axis, true); },
                Option::None => { break; }
            };
        };

        TensorTrait::<T>::new(shape, data)
    } else {
        TensorTrait::<T>::new(shape, data)
    }
}

/// Helper function that accumulates the mean of elements along a specific axis.
///
/// # Arguments
/// * `input_data` - The input's data.
/// * `input_shape` - The input's shape.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to accumulate the mean.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A value representing the accumulated mean along the specified axis.
fn accumulate_mean<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TDiv: Div<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut input_data: Span<T>, input_shape: Span<usize>, output_indices: Span<usize>, axis: usize
) -> T {
    let axis_len = *(input_shape)[axis];
    let mut acc: T = NumberTrait::zero();

    let mut axis_index: T = NumberTrait::zero();
    let mut axis_indexu32 = 0;

    if (input_shape).len() > 1 {
        while axis_indexu32 != axis_len {
            let input_indices = combine_indices(output_indices, axis_indexu32, axis);
            let input_index = ravel_index(input_shape, input_indices);
            let ele = *(input_data)[input_index];
            acc += ele;
            axis_index += NumberTrait::one();
            axis_indexu32 += 1;
        };
    } else {
        loop {
            match input_data.pop_front() {
                Option::Some(item) => {
                    acc += *item;
                    axis_index += NumberTrait::one();
                    axis_indexu32 += 1;
                },
                Option::None => { break; }
            };
        };
    }
    // let axis_index: T = NumberTrait::<T, MAG>::new(axis_index.try_into().unwrap(), false); 
    acc / axis_index
}
