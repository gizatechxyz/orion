use alexandria_data_structures::array_ext::ArrayTraitExt;
use alexandria_data_structures::span_ext::SpanTraitExt;
use alexandria_sorting::MergeSort;

use orion::numbers::{NumberTrait, U32IntoI32};
use orion::operators::tensor::core::{Tensor, TensorTrait, stride};
use orion::operators::tensor::helpers::{as_tensors_array, flatten_array_of_tensors};


/// Cf: TensorTrait::unique docstring
fn unique<
    T,
    +Copy<T>,
    +Drop<T>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +PartialEq<Tensor<T>>,
    +PartialOrd<Tensor<T>>
>(
    self: @Tensor<T>, axis: Option<usize>, sorted: Option<bool>
) -> (Tensor<T>, Tensor<i32>, Tensor<i32>, Tensor<i32>) {
    let sorted = match sorted {
        Option::Some(sorted) => sorted,
        Option::None => true,
    };

    let (unique_elements, new_shape, indices, inverse_indices, count) = if axis.is_none() {
        unique_flatten(self, sorted)
    } else {
        unique_along_axis(self, axis.unwrap(), sorted)
    };

    let unique_elements = Tensor::<T> { shape: new_shape, data: unique_elements };
    let indices = Tensor::<i32> { shape: array![indices.len()].span(), data: indices };
    let inverse_indices = Tensor::<
        i32
    > { shape: array![inverse_indices.len()].span(), data: inverse_indices };
    let count = Tensor::<i32> { shape: array![count.len()].span(), data: count };

    (unique_elements, indices, inverse_indices, count)
}

/// Subfunction unique for flatten tensors (no axis).
/// Cf: TensorTrait::unique docstring
fn unique_flatten<T, +Copy<T>, +Drop<T>, +PartialOrd<T>, +PartialEq<T>,>(
    t: @Tensor<T>, sorted: bool
) -> (Span<T>, Span<usize>, Span<i32>, Span<i32>, Span<i32>) {
    let mut indices: Array<i32> = array![];
    let mut inverse_indices: Array<i32> = array![];
    let mut count: Array<i32> = array![];

    let mut unique_elements = (*t.data).unique();
    let mut new_shape: Array<usize> = array![unique_elements.len()];

    if (sorted) {
        unique_elements = MergeSort::sort(unique_elements.span());
    }

    let mut unique_elements_span = unique_elements.span();
    let mut data_cpy = *(t.data);
    loop {
        match unique_elements_span.pop_front() {
            Option::Some(value) => {
                let occurences = data_cpy.occurrences(value);
                count.append(occurences.into());
                let idx_in_data = data_cpy.position(value).unwrap();
                indices.append(idx_in_data.into());
            },
            Option::None => { break; }
        }
    };
    unique_elements_span = unique_elements.span();
    loop {
        match data_cpy.pop_front() {
            Option::Some(value) => {
                let idx_in_uniques = unique_elements_span.position(value).unwrap();
                inverse_indices.append(idx_in_uniques.into());
            },
            Option::None => { break; }
        }
    };

    (unique_elements.span(), new_shape.span(), indices.span(), inverse_indices.span(), count.span())
}

/// Subfunction unique for tensors (wth axis).
/// Cf: TensorTrait::unique docstring
fn unique_along_axis<
    T,
    +Copy<T>,
    +Drop<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +TensorTrait<T>,
    +PartialEq<Tensor<T>>,
    +PartialOrd<Tensor<T>>
>(
    t: @Tensor<T>, axis: usize, sorted: bool
) -> (Span<T>, Span<usize>, Span<i32>, Span<i32>, Span<i32>) {
    let mut new_shape: Array<usize> = array![];
    let mut indices: Array<i32> = array![];
    let mut inverse_indices: Array<i32> = array![];
    let mut count: Array<i32> = array![];

    let rank = (*t).shape.len();
    assert(axis < rank, 'axis out of dimensions');

    let all_tensors = as_tensors_array(t, axis);
    let mut unique_tensors = all_tensors.unique();
    let mut unique_tensors_len = unique_tensors.len();

    let mut i = 0;
    while i != rank {
        new_shape.append(if axis == i {
            unique_tensors_len
        } else {
            *(*t).shape.at(i)
        });
        i += 1;
    };

    if (sorted) {
        unique_tensors = MergeSort::sort(unique_tensors.span());
    }

    let mut all_tensors_span = all_tensors.span();
    let mut unique_tensors_span = unique_tensors.span();
    loop {
        match unique_tensors_span.pop_front() {
            Option::Some(t) => {
                let occurences = all_tensors_span.occurrences(t);
                count.append(occurences.into());
                let idx_in_all = all_tensors_span.position(t).unwrap();
                indices.append(idx_in_all.into());
            },
            Option::None => { break; }
        }
    };
    unique_tensors_span = unique_tensors.span();
    loop {
        match all_tensors_span.pop_front() {
            Option::Some(t) => {
                let idx_in_uniques = unique_tensors_span.position(t).unwrap();
                inverse_indices.append(idx_in_uniques.into());
            },
            Option::None => { break; }
        }
    };

    let new_shape_span = new_shape.span();
    let unique_elements = flatten_array_of_tensors(unique_tensors, axis, new_shape_span);

    (unique_elements, new_shape_span, indices.span(), inverse_indices.span(), count.span())
}
