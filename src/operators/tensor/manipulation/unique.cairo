use core::traits::Into;
use core::traits::IndexView;
use option::OptionTrait;
use array::{SpanTrait, ArrayTrait};

use debug::PrintTrait;

use alexandria_data_structures::array_ext::SpanTraitExt;
use alexandria_sorting::merge_sort::merge;

use orion::numbers::{i32, NumberTrait};
use orion::operators::tensor::core::{Tensor, TensorTrait, stride};


/// Cf: TensorTrait::unique docstring
fn unique<
    T,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TPartialEqTensor: PartialEq<Tensor<T>>
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

    let unique_elements = Tensor::<T> { shape: new_shape.span(), data: unique_elements.span() };
    let indices = Tensor::<i32> { shape: array![indices.len()].span(), data: indices.span() };
    let inverse_indices = Tensor::<
        i32
    > { shape: array![inverse_indices.len()].span(), data: inverse_indices.span() };
    let count = Tensor::<i32> { shape: array![count.len()].span(), data: count.span() };

    (unique_elements, indices, inverse_indices, count)
}

/// Subfunction unique for flatten tensors (no axis).
fn unique_flatten<
    T,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
>(
    t: @Tensor<T>, sorted: bool
) -> (Array<T>, Array<usize>, Array<i32>, Array<i32>, Array<i32>) {
    let mut unique_elements: Array<T> = array![];
    let mut new_shape: Array<usize> = array![];
    let mut indices: Array<i32> = array![];
    let mut inverse_indices: Array<i32> = array![];
    let mut count: Array<i32> = array![];

    let mut data_cpy = *t.data;
    loop {
        match data_cpy.pop_front() {
            Option::Some(value) => {
                if !unique_elements.span().contains(*value) {
                    unique_elements.append(*value);
                }
            },
            Option::None => { break; }
        }
    };
    new_shape.append(unique_elements.len());

    // TODO: investigate why calling merge before the next 2 loops
    // cause the program to crash with error:
    // #73054: One of the arguments does not satisfy the requirements of the libfunc.
    if (sorted) {
        // unique_elements = merge(unique_elements);
        unique_elements = unique_elements;
    }

    let mut unique_elements_span = unique_elements.span();
    let mut data_cpy = *(t.data);
    loop {
        match unique_elements_span.pop_front() {
            Option::Some(value) => {
                let occurences = data_cpy.occurrences_of(*value);
                count.append(occurences.into());
                let idx_in_data = data_cpy.index_of(*value).unwrap();
                indices.append(idx_in_data.into());
            },
            Option::None => { break; }
        }
    };
    unique_elements_span = unique_elements.span();
    loop {
        match data_cpy.pop_front() {
            Option::Some(value) => {
                let idx_in_uniques = unique_elements_span.index_of(*value).unwrap();
                inverse_indices.append(idx_in_uniques.into());
            },
            Option::None => { break; }
        }
    };

    return (unique_elements, new_shape, indices, inverse_indices, count);
}

/// Subfunction unique for tensors (wth axis).
fn unique_along_axis<
    T,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TPartialEqTensor: PartialEq<Tensor<T>>
>(
    t: @Tensor<T>, axis: usize, sorted: bool
) -> (Array<T>, Array<usize>, Array<i32>, Array<i32>, Array<i32>) {
    let mut unique_tensors: Array<Tensor<T>> = array![];
    let mut new_shape: Array<usize> = array![];
    let mut indices: Array<i32> = array![];
    let mut inverse_indices: Array<i32> = array![];
    let mut count: Array<i32> = array![];

    let rank = (*t).shape.len();
    assert(axis <= rank, 'axis out of dimensions');

    let all_tensors = as_tensors_array(t, axis, rank);
    let mut unique_tensors = get_unique_tensors(all_tensors.span());
    let mut unique_tensors_len = unique_tensors.len();

    let mut i = 0;
    loop {
        if (i >= rank) {
            break;
        }
        new_shape.append(if axis == i {
            unique_tensors_len
        } else {
            *(*t).shape.at(i)
        });
        i += 1;
    };
    // TODO: need to implement PartialOrd for Tensor<T> in order to sort the tensors
    // using merge from Alexandria. (is this the right solution?)
    if (sorted) {
        // unique_tensors = merge(unique_tensors);
        unique_tensors = unique_tensors;
    }
    let mut unique_tensors_span = unique_tensors.span();
    let mut all_tensors_span = all_tensors.span();
    loop {
        match unique_tensors_span.pop_front() {
            Option::Some(t) => {
                let occurences = all_tensors_span.occurrences_of(*t);
                count.append(occurences.into());
                let idx_in_all = all_tensors_span.index_of(*t).unwrap();
                indices.append(idx_in_all.into());
            },
            Option::None => { break; }
        }
    };
    unique_tensors_span = unique_tensors.span();
    loop {
        match all_tensors_span.pop_front() {
            Option::Some(t) => {
                let idx_in_uniques = unique_tensors_span.index_of(*t).unwrap();
                inverse_indices.append(idx_in_uniques.into());
            },
            Option::None => { break; }
        }
    };

    let unique_elements = flatten_array_of_tensors(unique_tensors, axis, new_shape.span());

    return (unique_elements, new_shape, indices, inverse_indices, count);
}

/// Returns a Tensor as an array of tensors
fn as_tensors_array<
    T,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TPartialEqTensor: PartialEq<Tensor<T>>
>(
    tensor: @Tensor<T>, axis: usize, rank: usize
) -> Array<Tensor<T>> {
    let shape = *tensor.shape;
    let mut as_tensors: Array<Tensor<T>> = array![];

    let mut axes: Array<usize> = array![];
    let mut idx: usize = 0;
    loop {
        if idx >= rank {
            break;
        }
        axes.append(idx);
        idx += 1;
    };

    idx = 0;
    let axis_len: usize = *shape.at(axis);
    loop {
        if idx >= axis_len {
            break;
        }
        let mut starts: Array<usize> = array![];
        let mut ends: Array<usize> = array![];
        let mut i: usize = 0;
        loop {
            if i >= rank {
                break;
            }
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

/// TODO
fn get_unique_tensors<
    T,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TPartialEqTensor: PartialEq<Tensor<T>>
>(
    mut arr: Span<Tensor<T>>
) -> Array<Tensor<T>> {
    let mut uniques_tensors: Array<Tensor<T>> = array![];

    loop {
        match arr.pop_front() {
            Option::Some(t) => {
                if !uniques_tensors.span().contains(*t) {
                    uniques_tensors.append(*t);
                }
            },
            Option::None => { break; },
        }
    };
    return uniques_tensors;
}

/// TODO
fn flatten_array_of_tensors<
    T,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TPartialEqTensor: PartialEq<Tensor<T>>
>(
    tensors: Array<Tensor<T>>, axis: usize, new_shape: Span<usize>
) -> Array<T> {
    let num_sub_tensors = tensors.len(); // Number of sub-tensors in the array
    let mut new_stride = stride(new_shape);

    let mut tensors_span = tensors.span();
    let mut flattened: Array<T> = array![];
    loop {
        match tensors_span.pop_front() {
            Option::Some(mut t) => {
                let mut data = *t.data;
                loop {
                    match data.pop_front() {
                        Option::Some(v) => { flattened.append(*v); },
                        Option::None => { break; }
                    }
                };
            },
            Option::None => { break; },
        }
    };
    flattened
}
