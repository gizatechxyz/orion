use core::option::OptionTrait;
use core::traits::MulEq;
use array::ArrayTrait;
use array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};

/// reduce_prod - Reduces a tensor to its products along specified axis.
fn reduce_prod<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAddEq: AddEq<T>,
    impl TMulEq: MulEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, axis: usize, keepdims: bool
) -> Tensor<T> {
    /// Performs the reduction operation akin to ONNX's 'reduce_prod' on the given tensor for Orion Runtime.
    ///
    /// Given a tensor `self`, this function computes the product of elements along the specified axis.
    /// If the tensor is one-dimensional, the axis must be 0. For multi-dimensional tensors,
    /// the axis determines the dimension along which the reduction is performed.
    ///
    /// Arguments:
    /// - `self`: A reference to the input tensor on which the reduction operation is applied.
    /// - `axis`: The axis along which the reduction operation is performed.
    /// - `keepdims`: A boolean flag indicating whether to keep the reduced dimension in the output shape.
    ///
    /// Returns:
    /// - A new tensor resulting from the reduction operation.
    ///   If `keepdims` is `true`, the output tensor retains reduced dimensions;
    ///   otherwise, the reduced dimensions are eliminated from the output shape.
    ///
    /// # Panics
    /// - Panics if the specified axis is out of the tensor's dimensions.
    ///
    /// # Examples
    /// ```rust
    /// // Create a tensor
    /// let tensor_1 = TensorTrait::new(
    ///     shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),);
    ///
    /// // Reduce along axis 1 while keeping the dimension
    /// let reduced = reduce_prod(@tensor_1, 1, true);
    ///
    /// // Verify the shape of the reduced tensor
    /// assert(reduced.shape == array![2,1,2].span() , 'the tensors shapes are not equal');
    /// ```
    ///
    /// # Notes
    /// - This function utilizes accumulation of products along the specified axis to perform the reduction.
    /// - For one-dimensional tensors, the axis must be 0 to compute the product of all elements.
    ///
    /// # See Also
    /// - ONNX's 'reduce_prod' operation: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceProd 
    ///
    /// # References
    /// - Orion: https://orion.gizatech.xyz/
    /// - ONNX: Open Neural Network Exchange: https://onnx.ai/
    ///
    /// ```
    let mut output_data = ArrayTrait::new();

    if (*self.shape).len() == 1 {
        assert(axis == 0, 'axis out of dimensions');
        let current_prod = accumulate_production::<T>(*self.data, *self.shape, *self.shape, axis);
        output_data.append(current_prod);

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
            let current_sum = accumulate_production::<T>(*self.data, *self.shape, output_indices, axis);

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


/// Helper function that accumulates the product of elements along a specific axis.
///
/// # Arguments
/// * `input_data` - The input's data.
/// * `input_shape` - The input's shape.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to accumulate the product.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An i32 value representing the accumulated product along the specified axis.
fn accumulate_production<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAddEq: AddEq<T>,
    impl TMulEq: MulEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut input_data: Span<T>, input_shape: Span<usize>, output_indices: Span<usize>, axis: usize
) -> T {
    let axis_len = *(input_shape)[axis];
    let mut acc: T = NumberTrait::one();

    let mut axis_index: usize = 0;

    if (input_shape).len() > 1 {
        loop {
            if axis_index == axis_len {
                break ();
            }

            let input_indices = combine_indices(output_indices, axis_index, axis);
            let input_index = ravel_index(input_shape, input_indices);
            let ele = *(input_data)[input_index];
            acc *= ele;
            axis_index += 1;
        };
    } else {
        loop {
            match input_data.pop_front() {
                Option::Some(item) => { acc *= *item; },
                Option::None(_) => { break; }
            };
        };
    }

    return acc;
}
