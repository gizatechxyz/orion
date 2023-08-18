use array::{ArrayTrait, SpanTrait};
use serde::Serde;
use option::OptionTrait;


use orion::operators::tensor::helpers::{len_from_shape, check_shape};
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};

#[derive(Copy, Drop)]
struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>,
    extra: Option<ExtraParams>
}

#[derive(Serde, Copy, Drop)]
struct ExtraParams {
    fixed_point: Option<FixedImpl>
}

//Implement TensorSerde
impl TensorSerde<T, impl TSerde: Serde<T>, impl TDrop: Drop<T>> of Serde<Tensor<T>> {
    fn serialize(self: @Tensor<T>, ref output: Array<felt252>) {
        self.shape.serialize(ref output);
        self.data.serialize(ref output);
        self.extra.serialize(ref output);
    }

    fn deserialize(ref serialized: Span<felt252>) -> Option<Tensor<T>> {
        let shape: Span<usize> = Serde::<Span<usize>>::deserialize(ref serialized)?;
        let data: Span<T> = Serde::<Span<T>>::deserialize(ref serialized)?;
        let extra: Option<ExtraParams> = Serde::<Option<ExtraParams>>::deserialize(ref serialized)?;

        Option::Some(Tensor { shape, data, extra })
    }
}

/// Trait
///
/// new - Returns a new tensor with the given shape and data.
/// reshape - Returns a new tensor with the specified target shape and the same data as the input tensor.
/// flatten - Flattens the input tensor into a 2D tensor.
/// transpose - Returns a new tensor with the axes rearranged according to the given permutation.
/// at - Retrieves the value at the specified indices of a Tensor.
/// ravel_index - Converts a multi-dimensional index to a one-dimensional index.
/// unravel_index - Converts a one-dimensional index to a multi-dimensional index.
/// equal - Check if two tensors are equal element-wise.
/// greater - Check if each element of the first tensor is greater than the corresponding element of the second tensor.
/// greater_equal - Check if each element of the first tensor is greater than or equal to the corresponding element of the second tensor.
/// less - Check if each element of the first tensor is less than the corresponding element of the second tensor.
/// less_equal - Check if each element of the first tensor is less than or equal to the corresponding element of the second tensor.
/// or - Computes the logical OR of two tensors element-wise.
/// xor - Computes the logical XOR of two tensors element-wise.
/// stride - Computes the stride of each dimension in the tensor.
/// onehot - Produces one-hot tensor based on input.
/// min - Returns the minimum value in the tensor.
/// max - Returns the maximum value in the tensor.
/// reduce_sum - Reduces a tensor by summing its elements along a specified axis.
/// argmax - Returns the index of the maximum value along the specified axis.
/// argmin - Returns the index of the minimum value along the specified axis.
/// cumsum - Performs cumulative sum of the input elements along the given axis.
/// matmul - Performs matrix product of two tensors.
/// exp - Computes the exponential of all elements of the input tensor.
/// log - Computes the natural log of all elements of the input tensor.
/// abs - Computes the absolute value of all elements in the input tensor.
/// ceil - Rounds up the value of each element in the input tensor.
/// sqrt - Computes the square root of all elements of the input tensor.
/// sin - Computes the sine of all elements of the input tensor.
/// cos - Computes the cosine of all elements of the input tensor.
/// atan - Computes the arctangent (inverse of tangent) of all elements of the input tensor.
/// asin - Computes the arcsine (inverse of sine) of all elements of the input tensor.
/// acos - Computes the arccosine (inverse of cosine) of all elements of the input tensor.
/// sinh - Computes the hyperbolic sine of all elements of the input tensor.
/// tanh - Computes the hyperbolic tangent of all elements of the input tensor.
/// cosh - Computes the hyperbolic cosine of all elements of the input tensor.
/// asinh - Computes the inverse hyperbolic sine of all elements of the input tensor.
/// acosh - Computes the inverse hyperbolic cosine of all elements of the input tensor.
/// concat - Concatenate a list of tensors into a single tensor.
/// 
trait TensorTrait<T> {
    /// # tensor.new
    ///
    /// ```rust 
    ///    fn new(shape: Span<usize>, data: Span<T>, extra: Option<ExtraParams>) -> Tensor<T>;
    /// ```
    ///
    /// Returns a new tensor with the given shape and data.
    /// 
    /// ## Args
    /// 
    /// * `shape`(`Span<usize>`) - A span representing the shape of the tensor.
    /// * `data` (`Span<T>`) - A span containing the array of elements.
    /// * `extra` (`Option<ExtraParams>`) - A parameter for extra tensor options.
    ///
    /// ## Panics
    ///
    /// * Panics if the shape and data length are incompatible.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` instance.
    ///
    /// ## Examples
    /// 
    /// Let's create new u32 Tensors.
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// // 1D TENSOR
    /// fn tensor_1D() -> Tensor<u32> {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![3].span(),
    ///         data: array![0, 1, 2].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor;
    /// }
    /// 
    /// // 2D TENSOR
    /// fn tensor_2D() -> Tensor<u32> {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![2, 2].span(), 
    ///         data: array![0, 1, 2, 3].span(), 
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor;
    /// }
    /// 
    /// // 3D TENSOR
    /// fn tensor_3D() -> Tensor<u32> {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![2, 2, 2].span(), 
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(), 
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor;
    /// }
    /// ```
    ///
    fn new(shape: Span<usize>, data: Span<T>, extra: Option<ExtraParams>) -> Tensor<T>;
    /// # tensor.at
    ///
    /// ```rust 
    ///    fn at(self: @Tensor<T>, indices: Span<usize>) -> T;
    /// ```
    ///
    /// Retrieves the value at the specified indices of a Tensor.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `indices`(`Span<usize>`) - The indices to access element of the Tensor.
    ///
    /// ## Panics
    ///
    /// * Panics if the number of indices provided don't match the number of dimensions in the tensor.
    ///
    /// ## Returns
    ///
    /// The `T` value at the specified indices.
    ///
    /// # Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// 
    /// fn at_example() -> u32 {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `at` function as follows.
    ///     return tensor.at(
    ///         indices: array![0, 1, 1].span()
    ///     );
    /// }
    /// >>> 3
    /// ```
    /// 
    fn at(self: @Tensor<T>, indices: Span<usize>) -> T;
    /// # tensor.min
    ///
    /// ```rust 
    ///    fn min(self: @Tensor<T>) -> T;
    /// ```
    ///
    /// Returns the minimum value in the tensor.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// The minimum `T` value in the tensor.
    /// 
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// 
    /// fn min_example() -> u32 {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `min` function as follows.
    ///     return tensor.min();
    /// }
    /// >>> 0
    /// ```
    ///
    fn min(self: @Tensor<T>) -> T;
    /// # tensor.max
    ///
    /// ```rust 
    ///    fn max(self: @Tensor<T>) -> T;
    /// ```
    ///
    /// Returns the maximum value in the tensor.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// The maximum `T` value in the tensor.
    ///
    /// Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// 
    /// fn max_example() -> u32 {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `max` function as follows.
    ///     return tensor.max();
    /// }
    /// >>> 7
    /// ```
    /// 
    fn max(self: @Tensor<T>) -> T;
    /// # tensor.stride
    ///
    /// ```rust 
    ///    fn stride(self: @Tensor<T>) -> Span<usize>;
    /// ```
    ///
    /// Computes the stride of each dimension in the tensor.
    ///
    /// ## Args
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// A span of usize representing the stride for each dimension of the tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// 
    /// fn stride_example() -> Span<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `stride` function as follows.
    ///     return tensor.stride();
    /// }
    /// >>> [4,2,1]
    /// ```
    ///
    fn stride(self: @Tensor<T>) -> Span<usize>;
    /// # tensor.ravel_index
    ///
    /// ```rust 
    ///     fn ravel_index(self: @Tensor<T>, indices: Span<usize>) -> usize;
    /// ```
    ///
    /// Converts a multi-dimensional index to a one-dimensional index.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `indices`(`Span<usize>`) - The indices of the Tensor to ravel.
    ///
    /// ## Panics 
    ///
    /// * Panics if the indices are out of bounds of the Tensor shape.
    ///
    /// ## Returns
    /// 
    /// The index corresponding to the given indices.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// 
    /// fn ravel_index_example() -> usize {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `ravel_index` function as follows.
    ///     return tensor.ravel_index(
    ///         indices: array![1, 3, 0].span()
    ///     );
    /// }
    /// >>> 10 
    /// // This means that the value of indices [1,3,0] 
    /// // of a multidimensional array can be found at index 10 of Tensor.data.
    /// ```
    ///    
    fn ravel_index(self: @Tensor<T>, indices: Span<usize>) -> usize;
    /// # tensor.unravel_index
    ///
    /// ```rust 
    ///    fn unravel_index(self: @Tensor<T>, index: usize) -> Span<usize>;
    /// ```
    ///
    /// Converts a one-dimensional index to a multi-dimensional index.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `indices`(`Span<usize>`) - The index to unravel.
    ///
    /// ## Panics
    ///
    /// * Panics if the index is out of bounds of the Tensor shape.
    ///
    /// ## Returns
    ///
    /// The unraveled indices corresponding to the given index.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// 
    /// fn unravel_index_example() -> Span<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `unravel_index` function as follows.
    ///     return tensor.unravel_index(3);
    /// }
    /// >>> [0,1,1] 
    /// // This means that the value of index 3 of Tensor.data
    /// // can be found at indices [0,1,1] in multidimensional representation.
    /// ```
    ///
    fn unravel_index(self: @Tensor<T>, index: usize) -> Span<usize>;
    /// # tensor.reshape
    ///
    /// ```rust 
    ///    fn reshape(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T>;
    /// ```
    ///
    /// Returns a new tensor with the specified target shape and the same data as the input tensor.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `target_shape`(Span<usize>) - A span containing the target shape of the tensor.
    ///
    /// ## Panics
    ///
    /// * Panics if the target shape is incompatible with the input tensor's data.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` with the specified target shape and the same data.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// 
    /// fn reshape_tensor_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `reshape` function as follows.
    ///     return tensor.reshape(
    ///         target_shape: array![2,4].span()
    ///     );
    /// }
    /// >>> [[0,1,2,3], [4,5,6,7]]
    /// ```
    ///
    fn reshape(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T>;
    /// # tensor.transpose
    ///
    /// ```rust 
    ///    fn transpose(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T>;
    /// ```
    ///
    /// Returns a new tensor with the axes rearranged according to the given permutation.
    ///
    /// ## Args
    /// 
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `axes`(`Span<usize>`) - The usize elements representing the axes to be transposed.
    ///
    /// ## Panics
    ///
    /// * Panics if the length of the axes array is not equal to the rank of the input tensor.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` instance with the axes reordered according to the given permutation.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// 
    /// fn transpose_tensor_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `transpose` function as follows.
    ///     return tensor.transpose(
    ///         axes: array![1, 2, 0].span()
    ///     );
    /// }
    /// >>> [[[0,4],[1,5]],[[2,6],[3,7]]]
    /// ```
    ///
    fn transpose(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T>;
    /// ## tensor.reduce_sum
    ///
    /// ```rust 
    ///    fn reduce_sum(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// ```
    ///
    /// Reduces a tensor by summing its elements along a specified axis.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `axis`(`usize`) - The dimension to reduce.
    /// * `keepdims`(`bool`) - If true, retains reduced dimensions with length 1.
    ///
    /// ## Panics 
    /// 
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` instance with the specified axis reduced by summing its elements.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// 
    /// fn reduce_sum_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `reduce_sum` function as follows.
    ///     return tensor.reduce_sum(
    ///         axis: 0,
    ///         keepdims: false
    ///     );
    /// }
    /// >>> [[4,6],[8,10]]
    /// ```
    ///
    fn reduce_sum(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// # tensor.argmax
    ///
    /// ```rust 
    ///    fn argmax(self: @Tensor<T>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>) -> Tensor<usize>;
    /// ```
    ///
    /// Returns the index of the maximum value along the specified axis.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `axis`(`usize`) - The axis along which to compute the argmax.
    /// * `keepdims`(`Option<bool>`) - If true, retains reduced dimensions with length 1. Defaults to true.
    /// * `select_last_index`(`Option<bool>`) - If true, the index of the last occurrence of the maximum value is returned. Defaults to false.   
    ///
    /// ## Panics
    ///
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` instance containing the indices of the maximum values along the specified axis.
    ///
    /// ## Examples
    /// 
    /// Case 1: argmax with default parameters
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn argmax_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `argmax` function as follows.
    ///     return tensor.argmax(
    ///         axis: 2,
    ///         keepdims: Option::None(()),
    ///         select_last_index: Option::None(())
    ///     );
    /// }
    /// >>> [[[1,1],[0,0]]]
    /// ```
    /// Case 2: argmax with keepdims set to false
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn argmax_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `argmax` function as follows.
    ///     return tensor.argmax(
    ///         axis: 2,
    ///         keepdims: Option::Some(false),
    ///         select_last_index: Option::None(())
    ///     );
    /// }
    /// >>> [[1,1],[0,0]]
    /// ```
    ///
    /// Case 3: argmax with select_last_index set to true
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn argmax_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `argmax` function as follows.
    ///     return tensor.argmax(
    ///         axis: 2,
    ///         keepdims: Option::None(()),
    ///         select_last_index: Option::Some(true)
    ///     );
    /// }
    /// >>> [[[1,1],[1,1]]]
    /// ```
    ///
    fn argmax(
        self: @Tensor<T>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize>;
    /// # tensor.argmin
    ///
    /// ```rust 
    ///    fn argmin(self: @Tensor<T>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>) -> Tensor<usize>;
    /// ```
    ///
    /// Returns the index of the minimum value along the specified axis.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `axis`(`usize`) - The axis along which to compute the argmin.
    /// * `keepdims`(`Option<bool>`) - If true, retains reduced dimensions with length 1. Defaults to true.
    /// * `select_last_index`(`Option<bool>`) - If true, the index of the last occurrence of the minimum value is returned. Defaults to false.   
    ///
    /// ## Panics
    ///
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` instance containing the indices of the minimum values along the specified axis.
    ///
    /// ## Examples
    /// 
    /// Case 1: argmin with default parameters
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn argmin_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `argmin` function as follows.
    ///     return tensor.argmin(
    ///         axis: 2,
    ///         keepdims: Option::None(()),
    ///         select_last_index: Option::None(())
    ///     );
    /// }
    /// >>> [[[0,0],[0,0]]]
    ///
    /// ```
    /// Case 2: argmin with keepdims set to false
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn argmin_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `argmin` function as follows.
    ///     return tensor.argmin(
    ///         axis: 2,
    ///         keepdims: Option::Some(false),
    ///         select_last_index: Option::None(())
    ///     );
    /// }
    /// >>> [[0,0],[0,0]]
    /// ```
    ///
    /// Case 3: argmin with select_last_index set to true
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn argmin_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `argmin` function as follows.
    ///     return tensor.argmin(
    ///         axis: 2,
    ///         keepdims: Option::None(()),
    ///         select_last_index: Option::None(true)
    ///     );
    /// }
    /// >>> [[[0,0],[1,1]]]
    /// ```
    ///
    fn argmin(
        self: @Tensor<T>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize>;
    /// # tensor.matmul
    ///
    /// ```rust 
    ///    fn matmul(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Performs matrix product of two tensors.
    /// The behavior depends on the dimensionality of the tensors as follows:
    /// * If both tensors are 1-dimensional, the dot product is returned.
    /// * If both arguments are 2-dimensional, the matrix-matrix product is returned.
    /// * If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
    /// * If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - the first tensor to be multiplied
    /// * `other`(`@Tensor<T>`) - the second tensor to be multiplied
    ///
    /// ## Panics
    ///
    /// * Panics if the dimension of the tensors is higher than two.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` resulting from the matrix multiplication.
    ///
    /// ## Examples
    ///
    /// Case 1: Dot product of two vectors (1D \* 1D)
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn dot_product_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3].span(),
    ///         data: array![0, 1, 2].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3].span(),
    ///         data: array![0, 1, 2].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `matmul` function as follows.
    ///     return tensor_1.matmul(@tensor_2);
    /// }
    /// >>> [5]
    /// ```
    /// 
    /// Case 2: Matrix multiplication (2D \* 2D)
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn matrix_mul_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![244, 99, 109, 162].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![151, 68, 121, 170].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `matmul` function as follows.
    ///     return tensor_1.matmul(@tensor_2);
    /// }
    /// >>> [[48823, 33422],[36061, 34952]]
    /// ```
    /// 
    /// Case 3: Matrix-Vector multiplication (2D x 1D)
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn matrix_vec_mul_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3].span(),
    ///         data: array![0, 1, 2].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `matmul` function as follows.
    ///     return tensor_1.matmul(@tensor_2);
    /// }
    /// >>> [5,14,23]
    /// ```
    ///
    fn matmul(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
    /// # tensor.exp
    ///
    /// ```rust 
    ///     fn exp(self: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Computes the exponential of all elements of the input tensor.
    /// $$
    /// y_i=e^{x_i}
    /// $$
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// Returns a new tensor in `FixedType` with the exponential of the elements of the input tensor.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn exp_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![0, 1, 2, 3].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     // We can call `exp` function as follows.
    ///     return tensor.exp();
    /// }
    /// >>> [[8388608,22802594],[61983844,168489688]]
    /// // The fixed point representation of
    /// // [[1, 2.718281],[7.38905, 20.085536]]
    /// ```
    ///
    fn exp(self: @Tensor<T>) -> Tensor<FixedType>;
    /// # tensor.log
    ///
    /// ```rust 
    ///     fn log(self: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Computes the natural log of all elements of the input tensor.
    /// $$
    /// y_i=log({x_i})
    /// $$
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// Returns a new tensor in `FixedType` with the natural log of the elements of the input tensor.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn log_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![1, 2, 3, 100].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     // We can call `log` function as follows.
    ///     return tensor.log();
    /// }
    /// >>> [[0, 5814538, 9215825, 38630966]]
    /// // The fixed point representation of
    /// /// [[0, 0.693147, 1.098612, 4.605170]]
    /// ```
    ///
    fn log(self: @Tensor<T>) -> Tensor<FixedType>;
    /// #tensor.equal
    ///
    /// ```rust
    ///     fn equal(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Check if two tensors are equal element-wise.
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The first tensor to be equated
    /// * `other`(`@Tensor<T>`) - The second tensor to be equated
    ///
    /// ## Panics
    ///
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Returns
    ///
    /// A new `Tensor<usize>` of booleans (1 if equal, 0 otherwise) with the same shape as the broadcasted inputs.
    ///
    /// ## Examples
    ///
    /// Case 1: Compare tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn eq_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `equal` function as follows.
    ///     return tensor_1.equal(@tensor_2);
    /// }
    /// >>> [1,1,1,1,1,0,0,0]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn eq_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3].span(),
    ///         data: array![0, 1, 2].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `equal` function as follows.
    ///     return tensor_1.equal(@tensor_2);
    /// }
    /// >>> [1,1,1,0,0,0,0,0,0]
    /// ```
    ///
    fn equal(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// #tensor.greater
    ///
    /// ```rust
    ///     fn greater(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Check if each element of the first tensor is greater than the corresponding element of the second tensor.
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The first tensor to be compared
    /// * `other`(`@Tensor<T>`) - The second tensor to be compared
    ///
    /// ## Panics
    ///
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Returns
    ///
    /// A new `Tensor<usize>` of booleans (0 or 1) with the same shape as the broadcasted inputs.
    ///
    /// ## Examples
    ///
    /// Case 1: Compare tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn greater_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `greater` function as follows.
    ///     return tensor_1.greater(@tensor_2);
    /// }
    /// >>> [0,0,0,0,0,0,0,1,1]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn greater_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3].span(),
    ///         data: array![0, 1, 2].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `greater` function as follows.
    ///     return tensor_1.greater(@tensor_2);
    /// }
    /// >>> [0,0,0,1,1,1,1,1,1]
    /// ```
    ///
    fn greater(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// #tensor.greater_equal
    ///
    /// ```rust
    ///     fn greater_equal(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Check if each element of the first tensor is greater than or equal to the corresponding element of the second tensor.
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The first tensor to be compared
    /// * `other`(`@Tensor<T>`) - The second tensor to be compared
    ///
    /// ## Panics
    ///
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Returns
    ///
    /// A new `Tensor<usize>` of booleans (0 or 1) with the same shape as the broadcasted inputs.
    ///
    /// ## Examples
    ///
    /// Case 1: Compare tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn greater_equal_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `greater_equal` function as follows.
    ///     return tensor_1.greater_equal(@tensor_2);
    /// }
    /// >>> [1,1,1,1,1,1,0,1,1]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn greater_equal_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3].span(),
    ///         data: array![0, 1, 2].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `greater_equal` function as follows.
    ///     return tensor_1.greater_equal(@tensor_2);
    /// }
    /// >>> [1,1,1,1,1,1,0,0,0]
    /// ```
    ///
    fn greater_equal(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// #tensor.less
    ///
    /// ```rust
    ///     fn less(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Check if each element of the first tensor is less than the corresponding element of the second tensor.
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The first tensor to be compared
    /// * `other`(`@Tensor<T>`) - The second tensor to be compared
    ///
    /// ## Panics
    ///
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Returns
    ///
    /// A new `Tensor<usize>` of booleans (0 or 1) with the same shape as the broadcasted inputs.
    ///
    /// ## Examples
    ///
    /// Case 1: Compare tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn less_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `less` function as follows.
    ///     return tensor_1.less(@tensor_2);
    /// }
    /// >>> [0,0,0,0,0,0,1,0,0]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn less_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3].span(),
    ///         data: array![0, 1, 2].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `less` function as follows.
    ///     return tensor_1.less(@tensor_2);
    /// }
    /// >>> [0,0,0,0,0,0,0,1,1]
    /// ```
    ///
    fn less(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// #tensor.less_equal
    ///
    /// ```rust
    ///     fn less_equal(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Check if each element of the first tensor is less than or equal to the corresponding element of the second tensor.
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The first tensor to be compared
    /// * `other`(`@Tensor<T>`) - The second tensor to be compared
    ///
    /// ## Panics
    ///
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Returns
    ///
    /// A new `Tensor<usize>` of booleans (0 or 1) with the same shape as the broadcasted inputs.
    ///
    /// ## Examples
    ///
    /// Case 1: Compare tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn less_equal_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `less_equal` function as follows.
    ///     return tensor_1.less_equal(@tensor_2);
    /// }
    /// >>> [1,1,1,1,1,1,1,0,0]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// 
    /// fn less_equal_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(),
    ///         data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3].span(),
    ///         data: array![0, 1, 2].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     // We can call `less_equal` function as follows.
    ///     return tensor_1.less_equal(@tensor_2);
    /// }
    /// >>> [1,1,1,0,0,0,1,1,1]
    /// ```
    ///
    fn less_equal(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// #tensor.abs
    ///
    /// ```rust
    ///     fn abs(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the absolute value of all elements in the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the absolute value of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
    /// use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
    /// 
    /// fn abs_example() -> Tensor<i32> {
    ///     let tensor = TensorTrait::<i32>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             IntegerTrait::new(1, true), 
    ///             IntegerTrait::new(2, true), 
    ///             IntegerTrait::new(3, false)
    ///         ].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor.abs();
    /// }
    /// >>> [1, 2, 3]
    /// ```
    ///
    fn abs(self: @Tensor<T>) -> Tensor<T>;
    /// #tensor.ceil
    ///
    /// ```rust
    ///     fn ceil(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Rounds up the value of each element in the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the rounded up value of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn ceil_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new(29998, false), // 0.003576
    ///             FixedTrait::new(100663252, false), // 11.9999947548
    ///             FixedTrait::new(100663252, true) // -11.9999947548
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     return tensor.ceil();
    /// }
    /// >>> [1,12,-11]
    /// ```
    ///
    fn ceil(self: @Tensor<T>) -> Tensor<T>;
    /// #tensor.sin
    ///
    /// ```rust
    ///     fn sin(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the sine of all elements of the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the sine value of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn sin_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false)
    ///         ]
    ///             .span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     return tensor.sin();
    /// }
    /// >>> [0,7058770,7627740]
    /// // The fixed point representation of
    /// // [0,0.8414...,0.9092...]
    /// ```
    ///
    fn sin(self: @Tensor<T>) -> Tensor<FixedType>;
    /// #tensor.cos
    ///
    /// ```rust
    ///     fn cos(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the cosine of all elements of the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the cosine value of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn cos_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false)
    ///         ]
    ///             .span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     return tensor.cos();
    /// }
    /// >>> [8388608,4532384,-3490893]
    /// // The fixed point representation of
    /// // [1, 0.5403...,-0.4161]
    /// ```
    ///
    fn cos(self: @Tensor<T>) -> Tensor<FixedType>;
    /// # tensor.cumsum
    ///
    /// ```rust 
    ///    fn cumsum(self: @Tensor<T>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>) -> Tensor<usize>;
    /// ```
    ///
    /// Performs cumulative sum of the input elements along the given axis.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `axis`(`usize`) - The axis along which to compute the cumulative sum.
    /// * `exclusive`(`Option<bool>`) - By default, it will do the sum inclusively meaning the first element is copied as is.
    /// * `reverse`(`Option<bool>`) - If true, the cumulative sum is performed in the opposite direction. Defaults to false.   
    ///
    /// ## Panics
    ///
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` instance containing the cumulative sum of the input tensor's elements along the given axis.
    ///
    /// ## Examples
    /// 
    /// Case 1: cumsum with default parameters
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn cumsum_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor.cumsum(
    ///         axis: 2,
    ///         exclusive: Option::None(()),
    ///         reverse: Option::None(())
    ///     );
    /// }
    /// >>> [[[0,1],[2,5]],[[4,9],[6,13]]]
    /// ```
    ///
    /// Case 2: cumsum with exclusive = true
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn cumsum_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor.cumsum(
    ///         axis: 2,
    ///         exclusive: Option::Some(true),
    ///         reverse: Option::None(())
    ///     );
    /// }
    /// >>> [[[0,0],[0,2]],[[0,4],[0,6]]]
    /// ```
    ///
    /// Case 3: cumsum with exclusive = true and reverse = true
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn cumsum_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor.cumsum(
    ///         axis: 2,
    ///         exclusive: Option::Some(true),
    ///         reverse: Option::Some(true)
    ///     );
    /// }
    /// >>> [[[1,0],[3,0]],[[5,0],[7,0]]]
    /// ```
    ///
    fn cumsum(
        self: @Tensor<T>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<T>;
    /// # tensor.flatten
    ///
    /// ```rust 
    ///    fn flatten(self: @Tensor<T>, axis: usize) -> Tensor<T>;
    /// ```
    ///
    /// Flattens the input tensor into a 2D tensor. 
    /// If input tensor has shape (1, 2, 3,...n) then the output will have shape
    /// (1 * 2 * 3 * ... (axis-1), axis * (axis+1) * ... n).
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `axis`(`usize`) - Indicate up to which input dimensions (exclusive) should be flattened. 
    ///
    /// ## Panics
    ///
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` instance containing the flattened version of the input tensor.
    ///
    /// ## Examples
    /// 
    /// Case 1: flatten with axis 0
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn flatten_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor.flatten(0); // equivalent to tensor.reshape(1,8)
    /// }
    /// >>> [[0,1,2,5,4,9,6,13]]
    /// ```
    /// 
    /// Case 2: flatten with axis 1
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn flatten_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor.flatten(1); // equivalent to tensor.reshape(2,4)
    /// }
    /// >>> [[0,1,2,3],[4,5,6,7]]
    /// ```
    ///
    /// Case 3: flatten with axis 2
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn flatten_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor.flatten(2); // equivalent to tensor.reshape(4,2)
    /// }
    /// >>> [[0,1],[2,3],[4,5],[6,7]]
    /// ```
    ///
    fn flatten(self: @Tensor<T>, axis: usize) -> Tensor<T>;
    /// # tensor.sinh
    ///
    /// ```rust 
    ///     fn sinh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Computes the hyperbolic sine of all elements of the input tensor.
    /// $$
    /// y_i=sinh({x_i})
    /// $$
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// Returns a new tensor in `FixedType` with the hyperbolic sine of the elements of the input tensor.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn sinh_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![2,2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false)
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///    return tensor.sinh();
    /// }
    /// >>> [[0,9858303],[30424311,84036026]]
    /// // The fixed point representation of
    /// // [[0, 1.175201],[3.62686, 10.0178749]]
    /// ```
    ///
    fn sinh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// # tensor.tanh
    ///
    /// ```rust 
    ///     fn tanh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Computes the hyperbolic tangent of all elements of the input tensor.
    /// $$
    /// y_i=tanh({x_i})
    /// $$
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// Returns a new tensor in `FixedType` with the hyperbolic tangent of the elements of the input tensor.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn tanh_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![2,2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false)
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///    return tensor.tanh();
    /// }
    /// >>> [[0,6388715],[8086850,8347125]]
    /// // The fixed point representation of
    /// // [[0, 0.761594],[0.96403, 0.9951]]
    /// ```
    ///
    fn tanh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// # tensor.cosh
    ///
    /// ```rust 
    ///     fn cosh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Computes the hyperbolic cosine of all elements of the input tensor.
    /// $$
    /// y_i=cosh({x_i})
    /// $$
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// Returns a new tensor in `FixedType` with the hyperblic cosine of the elements of the input tensor.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn cosh_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![2,2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false)
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///    return tensor.cosh();
    /// }
    /// >>> [[8388608,12944299],[31559585,84453670]]
    /// // The fixed point representation of
    /// // [[, 1.54308],[3.762196, 10.067662]]
    /// ```
    ///
    fn cosh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// # tensor.asinh
    ///
    /// ```rust 
    ///     fn asinh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Computes the inverse hyperbolic sine of all elements of the input tensor.
    /// $$
    /// y_i=asinh({x_i})
    /// $$
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// Returns a new tensor in `FixedType` with the hyperblic sine of the elements of the input tensor.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn asinh_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![2,2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false)
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///    return tensor.asinh();
    /// }
    /// >>> [[0,7393498],[12110093,15254235]]
    /// // The fixed point representation of
    /// // [[0, 0.8814],[1.44364, 1.8185]]
    /// ```
    ///
    fn asinh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// # tensor.acosh
    ///
    /// ```rust 
    ///     fn acosh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Computes the inverse hyperbolic cosine of all elements of the input tensor.
    /// $$
    /// y_i=acosh({x_i})
    /// $$
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// Returns a new tensor in `FixedType` with the hyperblic cosine of the elements of the input tensor.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn acosh_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![2,2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false),
    ///             FixedTrait::new_unscaled(4, false)
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///    return tensor.acosh();
    /// }
    /// >>> [[0,11047444],[14786996,17309365]]
    /// // The fixed point representation of
    /// // [[0, 1.31696],[1.76275, 2.06344]]
    /// ```
    ///
    fn acosh(self: @Tensor<T>) -> Tensor<FixedType>;
    /// #tensor.atan
    ///
    /// ```rust
    ///     fn atan(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the arctangent (inverse of tangent) of all elements of the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the arctangent (inverse of tangent) value of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn atan_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///    return tensor.atan();
    /// }
    /// >>> [0,6588397,9287028]
    /// // The fixed point representation of
    /// // [0,0.7853...,1.1071...]
    /// ```
    ///    
    fn atan(self: @Tensor<T>) -> Tensor<FixedType>;
    /// #tensor.asin
    ///
    /// ```rust
    ///     fn asin(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the arcsine (inverse of sine) of all elements of the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the arcsine value of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn asin_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///    return tensor.asin();
    /// }
    /// >>> [0, 13176794]
    /// // The fixed point representation of
    /// // [0, 1.5707...]
    /// ```
    ///
    fn asin(self: @Tensor<T>) -> Tensor<FixedType>;
    /// #tensor.or
    ///
    /// ```rust
    ///     fn or(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Computes the logical OR of two tensors element-wise.
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The first tensor to be compared
    /// * `other`(`@Tensor<T>`) - The second tensor to be compared
    ///
    /// ## Panics
    ///
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Returns
    ///
    /// A new `Tensor<usize>` of booleans (0 or 1) with the same shape as the broadcasted inputs.
    ///
    /// ## Examples
    ///
    /// Case 1: Compare tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn or_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor_1.or(@tensor_2);
    /// }
    /// >>> [0,1,1,1,1,1,1,1,1]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn or_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![1, 3].span(), 
    ///         data: array![0, 1, 2].span(), 
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor_1.or(@tensor_2);
    /// }
    /// >>> [0,1,1,1,1,1,1,1,1]
    /// ```
    ///
    fn or(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// #tensor.xor
    ///
    /// ```rust
    ///     fn xor(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Computes the logical XOR of two tensors element-wise.
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The first tensor to be compared
    /// * `other`(`@Tensor<T>`) - The second tensor to be compared
    ///
    /// ## Panics
    ///
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Returns
    ///
    /// A new `Tensor<usize>` of booleans (0 or 1) with the same shape as the broadcasted inputs.
    ///
    /// ## Examples
    ///
    /// Case 1: Compare tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn xor_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor_1.xor(@tensor_2);
    /// }
    /// >>> [0,0,0,0,0,0,0,0,0]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn xor_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![1, 3].span(), 
    ///         data: array![0, 1, 2].span(), 
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor_1.xor(@tensor_2);
    /// }
    /// >>> [0,0,0,1,0,0,1,0,0]
    /// ```
    ///
    fn xor(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// #tensor.acos
    ///
    /// ```rust
    ///     fn acos(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the arccosine (inverse of cosine) of all elements of the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the arccosine value of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn acos_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///    return tensor.acos();
    /// }
    /// >>> [13176794, 0]
    /// // The fixed point representation of
    /// // [1.5707..., 0]
    /// ```
    ///
    fn acos(self: @Tensor<T>) -> Tensor<FixedType>;
    /// # tensor.onehot
    ///
    /// ```rust 
    ///    fn onehot(self: @Tensor<T>, depth: usize, axis: Option<usize>, values: Span<usize>) -> Tensor<usize>;
    /// ```
    ///
    /// Produces one-hot tensor based on input.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `depth`(`usize`) - Scalar or Rank 1 tensor containing exactly one element, specifying the number of classes in one-hot tensor.
    /// * `axis`(`Option<bool>`) - Axis along which one-hot representation in added. Default: axis=-1.
    /// * `values`(`Span<usize>`) - Rank 1 tensor containing exactly two elements, in the format [off_value, on_value]   
    ///
    /// ## Panics
    ///
    /// * Panics if values is not equal to 2.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` one-hot encode of the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// fn onehot_example() -> Tensor<FixedType> {
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// 
    /// fn onehot_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(), 
    ///         data: array![0, 1, 2, 3].span(), 
    ///         extra: Option::None(())
    ///     );
    /// 
    ///     return tensor.onehot(
    ///         depth: 3, 
    ///         axis: Option::None(()), 
    ///         values: array![0, 1].span()
    ///     );
    /// }
    /// >>> [[1. 0. 0.]
    ///      [0. 1. 0.]
    ///      [0. 0. 1.]]
    /// ```
    ///
    fn onehot(
        self: @Tensor<T>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<T>;
    /// #tensor.sqrt
    ///
    /// ```rust
    ///     fn sqrt(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the square root of all elements of the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the arctangent (inverse of tangent) value of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
    /// use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn sqrt_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<FixedType>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     return tensor.sqrt();
    /// }
    /// >>> [0,8388608,11863169]
    /// // The fixed point representation of
    /// // [0,1,1.4142...]
    /// ```
    ///    
    fn sqrt(self: @Tensor<T>) -> Tensor<FixedType>;   
    /// # tensor.concat
    ///
    /// ```rust 
    ///    fn concat(tensors: Span<Tensor<T>>, axis: usize,  ) -> Tensor<T>;
    /// ```
    ///
    /// Concatenate a list of tensors into a single tensor..
    ///
    /// ## Args
    ///
    /// * `tensors`(` Span<Tensor<T>>,`) - Array of the input tensors.
    /// * `axis`(`usize`) -  Axis to concat on.
    ///
    /// ## Panics
    ///
    /// * Panics if lenght of tensors is not equal greater than 1.
    /// * Panics if dimension is not greater than axis
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` concatenated tensor of the input tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// fn concat_example() -> Tensor<FixedType> {
    ///    let tensor1 = u32_tensor_2x2_helper();
    ///    let tensor2 = u32_tensor_2x2_helper();
    ///
    ///    let mut data = ArrayTrait::new();
    ///    data.append(tensor1);
    ///    data.append(tensor2);
    ///    axis = 0 
    ///    let result = TensorTrait::concat(tensors: data.span(), axis: axis);
    ///    return result;
    /// }
    /// >>> [[0. 1.]
    ///      [2. 3.],
    ///      [0. 1.]
    ///      [2. 3.]]
    ///
    ///     result.shape
    /// >>> (4, 2)
    ///
    ///    axis = 1
    ///    let result = TensorTrait::concat(tensors: data.span(), axis: axis);
    ///    return result;
    /// }
    /// >>> [[0. 1., 0., 1.]
    ///      [2. 3., 2., 3.]]
    ///
    ///     result.shape
    /// >>> (2, 4 ) 
    /// ```
    ///
    fn concat(tensors: Span<Tensor<T>>, axis: usize,  ) -> Tensor<T>;
}


/// Cf: TensorTrait::new docstring
fn new_tensor<T>(shape: Span<usize>, data: Span<T>, extra: Option<ExtraParams>) -> Tensor<T> {
    check_shape::<T>(shape, data);
    Tensor::<T> { shape, data, extra }
}

/// Cf: TensorTrait::ravel_index docstring
fn ravel_index(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    assert(shape.len() == indices.len(), 'shape & indices length unequal');

    let mut raveled_index: usize = 0;
    let mut stride: usize = 1;

    loop {
        if shape.len() == 0 {
            break ();
        }

        let index = *indices.pop_back().unwrap();
        raveled_index += index * stride;

        stride *= *shape.pop_back().unwrap();
    };

    raveled_index
}

/// Cf: TensorTrait::unravel_index docstring
fn unravel_index(index: usize, mut shape: Span<usize>) -> Span<usize> {
    assert(shape.len() > 0, 'shape cannot be empty');

    let mut result = ArrayTrait::new();
    let mut remainder = index;
    let mut stride = len_from_shape(shape);

    loop {
        if shape.len() == 0 {
            break ();
        }

        stride /= *shape.pop_front().unwrap();

        let coord = remainder / stride;
        remainder = remainder % stride;

        result.append(coord);
    };

    return result.span();
}

/// Cf: TensorTrait::stride docstring
fn stride(mut shape: Span<usize>) -> Span<usize> {
    let shape_len = shape.len();
    assert(shape_len > 0, 'shape cannot be empty');

    let mut result: Array<usize> = ArrayTrait::new();
    let mut accumulated: usize = 1;
    let mut temp_result = ArrayTrait::new();
    loop {
        temp_result.append(accumulated);

        if shape.len() == 0 {
            break ();
        }
        accumulated *= *shape.pop_back().unwrap();
    };

    let mut i: usize = shape_len - 1;
    loop {
        result.append(*temp_result.at(i));

        if i == 0 {
            break ();
        }
        i -= 1;
    };

    return result.span();
}

/// Cf: TensorTrait::reshape docstring
fn reshape<T>(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T> {
    new_tensor(target_shape, *self.data, *self.extra)
}

/// Cf: TensorTrait::at docstring
fn at_tensor<T>(self: @Tensor<T>, indices: Span<usize>) -> @T {
    assert(indices.len() == (*self.shape).len(), 'indices not match dimensions');
    let data = *self.data;

    return data.at(ravel_index(*self.shape, indices));
}

// Return true if two tensor are equal
fn tensor_eq<T, impl TPartialEq: PartialEq<T>>(mut lhs: Tensor<T>, mut rhs: Tensor<T>, ) -> bool {
    let mut is_eq = true;

    loop {
        if lhs.shape.len() == 0 || !is_eq {
            break;
        }

        is_eq = lhs.shape.pop_front().unwrap() == rhs.shape.pop_front().unwrap();
    };

    if !is_eq {
        return false;
    }

    loop {
        if lhs.data.len() == 0 || !is_eq {
            break;
        }

        is_eq = lhs.data.pop_front().unwrap() == rhs.data.pop_front().unwrap();
    };

    return is_eq;
}

