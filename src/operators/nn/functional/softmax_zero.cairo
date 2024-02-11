use core::traits::Into;
use core::option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};


/// Cf: NNTrait::softmax_zero docstring
fn softmax_zero<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TPartialEq: PartialEq<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TAddEq: AddEq<T>,
>(
    z: @Tensor<T>, axis: usize
) -> Tensor<T> {
    let exp_tensor = exp_zero(*z);
    let sum_no_zero = reduce_sum_no_zero(@exp_tensor, axis, true);
    exp_tensor / sum_no_zero
}

/// Cf: NNTrait::softmax_zero docstring
fn softmaxWide_zero<
    T,
    TMAG,
    W,
    WMAG,
    impl TTensor: TensorTrait<T>,
    impl WTensor: TensorTrait<W>,
    impl TDiv: Div<T>,
    impl TIntoW: Into<T, W>,
    impl WTryIntoT: TryInto<W, T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl WCopy: Copy<W>,
    impl WDrop: Drop<W>,
    impl TNumber: NumberTrait<T, TMAG>,
    impl WNumber: NumberTrait<W, WMAG>,
    impl TPartialEq: PartialEq<T>,
    impl WPartialEq: PartialEq<W>,
    impl TAddEq: AddEq<T>,
    impl WAddEq: AddEq<W>,
>(
    z: @Tensor<T>, axis: usize
) -> Tensor<T> {
    let exp_tensor: Tensor<W> = exp_upcast_zero(*z);
    let sum_no_zero = reduce_sum_no_zero(@exp_tensor, axis, true);
    div_downcast(@exp_tensor, @sum_no_zero)
}


/// Helper function that compute the exponential of a tensor except if the value of an entry is zero, the value remains zero.
///
/// # Arguments
/// * `z` - The input tensor.
///
/// # Returns
/// * A Tensor<T> representing the exponential of the tensor except for the entries equal to zero in the input tensor, they remain zero.
fn exp_zero<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl FTensor: TensorTrait<T>,
    impl TPartialEq: PartialEq<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    mut z: Tensor<T>
) -> Tensor<T> {
    let mut result = ArrayTrait::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                if *item == NumberTrait::zero() {
                    result.append(NumberTrait::zero());
                } else {
                    result.append((*item).exp());
                }
            },
            Option::None => { break; }
        };
    };

    return TensorTrait::new(z.shape, result.span());
}

/// Helper function that compute the exponential of a tensor except if the value of an entry is zero, the value remains zero.
///
/// # Arguments
/// * `z` - The input tensor.
///
/// # Returns
/// * A Tensor<T> representing the exponential of the tensor except for the entries equal to zero in the input tensor, they remain zero.
fn exp_upcast_zero<
    T,
    TMAG,
    W,
    WMAG,
    impl TNumber: NumberTrait<T, TMAG>,
    impl TTensor: TensorTrait<T>,
    impl TPartialEq: PartialEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl WNumber: NumberTrait<W, WMAG>,
    impl WTensor: TensorTrait<W>,
    impl WCopy: Copy<W>,
    impl WDrop: Drop<W>,
    impl TIntoW: Into<T, W>,
>(
    mut self: Tensor<T>
) -> Tensor<W> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                if *item == NumberTrait::zero() {
                    result.append(NumberTrait::zero());
                } else {
                    result.append((TIntoW::into(*item)).exp());
                }
            },
            Option::None => { break; }
        };
    };

    return TensorTrait::new(self.shape, result.span());
}


/// Helper function that compute the reduce sum making sure no none zero value are in the output tensor.
///
/// # Arguments
/// * `z` - The input tensor.
///
/// # Returns
/// * A Tensor<T> representing the ereduce sum with no entries equal to zero.

fn reduce_sum_no_zero<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TPartialEq: PartialEq<T>,
>(
    self: @Tensor<T>, axis: usize, keepdims: bool
) -> Tensor<T> {
    let mut output_data = ArrayTrait::new();

    if (*self.shape).len() == 1 {
        assert(axis == 0, 'axis out of dimensions');
        let current_sum = accumulate_sum::<T>(*self.data, *self.shape, *self.shape, axis);
        output_data.append(current_sum);

        let mut output_shape = ArrayTrait::new();
        output_shape.append(1);

        return TensorTrait::new(output_shape.span(), output_data.span());
    } else {
        assert(axis <= (*self.shape).len(), 'axis out of dimensions');
        let output_shape = reduce_output_shape(*self.shape, axis, false);
        let output_data_len = len_from_shape(output_shape);
        let mut index: usize = 0;
        loop {
            let output_indices = unravel_index(index, output_shape);
            let mut current_sum = accumulate_sum::<T>(*self.data, *self.shape, output_indices, axis);

            if current_sum == NumberTrait::zero() {
                current_sum = NumberTrait::one();
            }
            output_data.append(current_sum);

            index += 1;
            if index == output_data_len {
                break ();
            };
        };

        if keepdims {
            let output_shape = reduce_output_shape(*self.shape, axis, true);
            return TensorTrait::<T>::new(output_shape, output_data.span());
        } else {
            return TensorTrait::<T>::new(output_shape, output_data.span());
        }
    }
}
