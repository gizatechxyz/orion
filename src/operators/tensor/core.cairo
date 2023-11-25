use array::{ArrayTrait, SpanTrait};
use serde::Serde;
use option::OptionTrait;

use alexandria_data_structures::array_ext::{SpanTraitExt};

use orion::operators::tensor::helpers::{len_from_shape, check_shape};
use orion::numbers::{i8, i32, NumberTrait};

#[derive(Copy, Drop)]
struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>,
}

//Implement TensorSerde
impl TensorSerde<T, impl TSerde: Serde<T>, impl TDrop: Drop<T>> of Serde<Tensor<T>> {
    fn serialize(self: @Tensor<T>, ref output: Array<felt252>) {
        self.shape.serialize(ref output);
        self.data.serialize(ref output);
    }

    fn deserialize(ref serialized: Span<felt252>) -> Option<Tensor<T>> {
        let shape: Span<usize> = Serde::<Span<usize>>::deserialize(ref serialized)?;
        let data: Span<T> = Serde::<Span<T>>::deserialize(ref serialized)?;

        Option::Some(Tensor { shape, data })
    }
}

/// Trait
///
/// new - Returns a new tensor with the given shape and data.
/// reshape - Returns a new tensor with the specified target shape and the same data as the input tensor.
/// flatten - Flattens the input tensor into a 2D tensor.
/// constant_of_shape - Generate a tensor with given value and shape.
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
/// max_in_tensor - Returns the maximum value in the tensor.
/// min_in_tensor - Returns the minimum value in the tensor.
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
/// neg - Computes the negation of all elements in the input tensor.
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
/// slice - Produces a slice of the input tensor along multiple axes. 
/// concat - Concatenate a list of tensors into a single tensor.
/// quantize_linear - Quantizes a Tensor to i8 using linear quantization.
/// dequantize_linear - Dequantizes an i8 Tensor using linear dequantization.
/// qlinear_add - Performs the sum of two quantized i8 Tensors.
/// qlinear_mul - Performs the element-wise multiplication of quantized Tensors.
/// qlinear_matmul - Performs the product of two quantized i8 Tensors.
/// qlinear_concat - Concatenate a list of tensors after dequantizing them with their respective scales and zero_points and returns the quantized result.
/// gather - Gather entries of the axis dimension of data.
/// nonzero - Produces indices of the elements that are non-zero (in row-major order - by dimension).
/// squeeze - Removes dimensions of size 1 from the shape of a tensor.
/// unsqueeze - Inserts single-dimensional entries to the shape of an input tensor.
/// sign - Calculates the sign of the given input tensor element-wise.
/// clip - Clip operator limits the given input within an interval.
/// and - Computes the logical AND of two tensors element-wise. 
/// identity - Return a Tensor with the same shape and contents as input.
/// where - Return elements chosen from x or y depending on condition.
/// bitwise_and - Computes the bitwise AND of two tensors element-wise.
/// round - Computes the round value of all elements in the input tensor.
/// reduce_l1 - Computes the L1 norm of the input tensor's elements along the provided axes.
/// trilu - Returns the upper or lower triangular part of a tensor or a batch of 2D matrices.
/// scatter - Produces a copy of input data, and updates value to values specified by updates at specific index positions specified by indices.
/// reduce_sum_square - Computes the sum square of the input tensor's elements along the provided axes. 
/// reduce_l2 - Computes the L2 norm of the input tensor's elements along the provided axes.
/// sequence_length - Returns the length of the input sequence.
/// sequence_insert - Insert a tensor into a sequence.
/// sequence_at - Outputs the tensor at the specified position in the input sequence.
/// sequence_construct - Constructs a tensor sequence containing the input tensors.
/// shrink - Shrinks the input tensor element-wise to the output tensor.
/// reduce_mean - Computes the mean of the input tensor's elements along the provided axes.
/// pow - Pow takes input data (Tensor) and exponent Tensor, and produces one output data (Tensor) where the function f(x) = x^exponent, is applied to the data tensor elementwise.
/// sequence_erase – Outputs the tensor sequence with the erased tensor at the specified position.
/// sequence_empty - Returns an empty tensor sequence.
/// binarizer - Maps the values of a tensor element-wise to 0 or 1 based on the comparison against a threshold value.
/// array_feature_extractor - Selects elements of the input tensor based on the indices passed applied to the last tensor axis.
/// reduce_min - Computes the min of the input tensor's elements along the provided axes.
/// 
trait TensorTrait<T> {
    /// # tensor.new
    ///
    /// ```rust 
    ///    fn new(shape: Span<usize>, data: Span<T>) -> Tensor<T>;
    /// ```
    ///
    /// Returns a new tensor with the given shape and data.
    /// 
    /// ## Args
    /// 
    /// * `shape`(`Span<usize>`) - A span representing the shape of the tensor.
    /// * `data` (`Span<T>`) - A span containing the array of elements.
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
    /// use orion::operators::tensor::{
    ///     TensorTrait, // we import the trait
    ///     Tensor, // we import the type
    ///     U32Tensor // we import the implementation. 
    /// };
    /// 
    /// // 1D TENSOR
    /// fn tensor_1D() -> Tensor<u32> {
    ///     let tensor = TensorTrait::new(shape: array![3].span(), data: array![0, 1, 2].span());
    /// 
    ///     return tensor;
    /// }
    /// 
    /// // 2D TENSOR
    /// fn tensor_2D() -> Tensor<u32> {
    ///     let tensor = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    /// 
    ///     return tensor;
    /// }
    /// 
    /// // 3D TENSOR
    /// fn tensor_3D() -> Tensor<u32> {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     return tensor;
    /// }
    /// ```
    ///
    fn new(shape: Span<usize>, data: Span<T>) -> Tensor<T>;
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// 
    /// fn at_example() -> u32 {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `at` function as follows.
    ///     return tensor.at(indices: array![0, 1, 1].span());
    /// }
    /// >>> 3
    /// ```
    /// 
    fn at(self: @Tensor<T>, indices: Span<usize>) -> T;
    /// # tensor.min_in_tensor
    ///
    /// ```rust 
    ///    fn min_in_tensor(self: @Tensor<T>) -> T;
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn min_in_tensor_example() -> u32 {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `min_in_tensor` function as follows.
    ///     return tensor.min_in_tensor();
    /// }
    /// >>> 0
    /// ```
    ///
    fn min_in_tensor(self: @Tensor<T>) -> T;
    /// # tensor.min
    ///
    /// ```rust 
    ///    fn min(tensors: Span<Tensor<T>>) -> Tensor<T>;
    /// ```
    ///
    /// Returns the element-wise minumum values from a list of input tensors
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    ///
    /// ## Args
    ///
    /// * `tensors`(` Span<Tensor<T>>,`) - Array of the input tensors
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` containing the element-wise minimum values
    ///
    /// ## Panics
    ///
    /// * Panics if tensor array is empty
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Examples
    ///
    /// Case 1: Process tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn min_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 3, 1, 2].span(),);
    ///     let result = TensorTrait::min(tensors: array![tensor1, tensor2].span());
    ///     return result;
    /// }
    /// >>> [0, 1, 1, 2]
    ///
    ///     result.shape
    /// >>> (2, 2)
    /// ```
    /// 
    /// Case 2: Process tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn min_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    ///     let tensor2 = TensorTrait::new(shape: array![1, 2].span(), data: array![1, 4].span(),);
    ///     let result = TensorTrait::min(tensors: array![tensor1, tensor2].span());
    ///     return result;
    /// }
    /// >>> [0, 1, 1, 4]
    ///
    ///     result.shape
    /// >>> (2, 2)
    /// ```
    ///
    fn min(tensors: Span<Tensor<T>>) -> Tensor<T>;
    /// # tensor.max_in_tensor
    ///
    /// ```rust 
    ///    fn max_in_tensor(self: @Tensor<T>) -> T;
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn max_in_tensor_example() -> u32 {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `max_in_tensor` function as follows.
    ///     return tensor.max_in_tensor();
    /// }
    /// >>> 7
    /// ```
    /// 
    fn max_in_tensor(self: @Tensor<T>) -> T;
    /// # tensor.max
    ///
    /// ```rust 
    ///    fn max(tensors: Span<Tensor<T>>) -> Tensor<T>;
    /// ```
    ///
    /// Returns the element-wise maximum values from a list of input tensors
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    ///
    /// ## Args
    ///
    /// * `tensors`(` Span<Tensor<T>>,`) - Array of the input tensors
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` containing the element-wise maximum values
    ///
    /// ## Panics
    ///
    /// * Panics if tensor array is empty
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Examples
    ///
    /// Case 1: Process tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn max_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 3, 1, 2].span(),);
    ///     let result = TensorTrait::max(tensors: array![tensor1, tensor2].span());
    ///     return result;
    /// }
    /// >>> [0, 3, 2, 3]
    ///
    ///     result.shape
    /// >>> (2, 2)
    /// ```
    /// 
    /// Case 2: Process tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn max_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    ///     let tensor2 = TensorTrait::new(shape: array![1, 2].span(), data: array![1, 4].span(),);
    ///     let result = TensorTrait::max(tensors: array![tensor1, tensor2].span());
    ///     return result;
    /// }
    /// >>> [1, 4, 2, 4]
    ///
    ///     result.shape
    /// >>> (2, 2)
    /// ```
    ///
    fn max(tensors: Span<Tensor<T>>) -> Tensor<T>;
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn stride_example() -> Span<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn ravel_index_example() -> usize {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `ravel_index` function as follows.
    ///     return tensor.ravel_index(indices: array![1, 3, 0].span());
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn unravel_index_example() -> Span<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn reshape_tensor_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `reshape` function as follows.
    ///     return tensor.reshape(target_shape: array![2, 4].span());
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn transpose_tensor_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `transpose` function as follows.
    ///     return tensor.transpose(axes: array![1, 2, 0].span());
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn reduce_sum_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `reduce_sum` function as follows.
    ///     return tensor.reduce_sum(axis: 0, keepdims: false);
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn argmax_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///     );
    /// 
    ///     // We can call `argmax` function as follows.
    ///     return tensor.argmax(axis: 2, keepdims: Option::None(()), select_last_index: Option::None(()));
    /// }
    /// >>> [[[1,1],[0,0]]]
    /// ```
    /// Case 2: argmax with keepdims set to false
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn argmax_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///     );
    /// 
    ///     // We can call `argmax` function as follows.
    ///     return tensor
    ///         .argmax(axis: 2, keepdims: Option::Some(false), select_last_index: Option::None(()));
    /// }
    /// >>> [[1,1],[0,0]]
    /// ```
    ///
    /// Case 3: argmax with select_last_index set to true
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn argmax_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///     );
    /// 
    ///     // We can call `argmax` function as follows.
    ///     return tensor
    ///         .argmax(axis: 2, keepdims: Option::None(()), select_last_index: Option::Some(true));
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn argmin_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///     );
    /// 
    ///     // We can call `argmin` function as follows.
    ///     return tensor.argmin(axis: 2, keepdims: Option::None(()), select_last_index: Option::None(()));
    /// }
    /// >>> [[[0,0],[0,0]]]
    ///
    /// ```
    /// Case 2: argmin with keepdims set to false
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn argmin_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///     );
    /// 
    ///     // We can call `argmin` function as follows.
    ///     return tensor
    ///         .argmin(axis: 2, keepdims: Option::Some(false), select_last_index: Option::None(()));
    /// }
    /// >>> [[0,0],[0,0]]
    /// ```
    ///
    /// Case 3: argmin with select_last_index set to true
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn argmin_example() -> Tensor<usize> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 4, 5, 5].span(),
    ///     );
    /// 
    ///     // We can call `argmin` function as follows.
    ///     return tensor
    ///         .argmin(axis: 2, keepdims: Option::None(()), select_last_index: Option::Some(true));
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn dot_product_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn matrix_mul_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(), data: array![244, 99, 109, 162].span()
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(), data: array![151, 68, 121, 170].span()
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn matrix_vec_mul_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);
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
    ///     fn exp(self: @Tensor<T>) -> Tensor<T>;
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
    /// Returns a new tensor in `T` with the exponential of the elements of the input tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn exp_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(), 
    ///         data: array![
    ///                 FixedTrait::new_unscaled(0, false), 
    ///                 FixedTrait::new_unscaled(1, false), 
    ///                 FixedTrait::new_unscaled(2, false), 
    ///                 FixedTrait::new_unscaled(3, false), 
    ///             ]
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
    fn exp(self: @Tensor<T>) -> Tensor<T>;
    /// # tensor.log
    ///
    /// ```rust 
    ///     fn log(self: @Tensor<T>) -> Tensor<T>;
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
    /// Returns a new tensor in `T` with the natural log of the elements of the input tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn log_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(), 
    ///         data: array![
    ///                 FixedTrait::new_unscaled(0, false), 
    ///                 FixedTrait::new_unscaled(1, false), 
    ///                 FixedTrait::new_unscaled(2, false), 
    ///                 FixedTrait::new_unscaled(100, false), 
    ///             ]
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
    fn log(self: @Tensor<T>) -> Tensor<T>;
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn eq_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn eq_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn greater_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn greater_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn greater_equal_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn greater_equal_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn less_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn less_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn less_equal_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn less_equal_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
    /// use orion::numbers::{i32, IntegerTrait};
    /// 
    /// fn abs_example() -> Tensor<i32> {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             IntegerTrait::new(1, true), IntegerTrait::new(2, true), IntegerTrait::new(3, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.abs();
    /// }
    /// >>> [1, 2, 3]
    /// ```
    ///
    fn abs(self: @Tensor<T>) -> Tensor<T>;
    /// #tensor.neg
    ///
    /// ```rust
    ///     fn neg(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the negation of all elements in the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the negation of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
    /// use orion::numbers::{i32, IntegerTrait};
    /// 
    /// fn neg_example() -> Tensor<i32> {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             IntegerTrait::new(1, true), IntegerTrait::new(2, true), IntegerTrait::new(3, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.neg();
    /// }
    /// >>> [1, 2, -3]
    /// ```
    ///
    fn neg(self: @Tensor<T>) -> Tensor<T>;
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn ceil_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new(29998, false), // 0.003576
    ///             FixedTrait::new(100663252, false), // 11.9999947548
    ///             FixedTrait::new(100663252, true) // -11.9999947548
    ///         ]
    ///             .span(),
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn sin_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.sin();
    /// }
    /// >>> [0,7058770,7627740]
    /// // The fixed point representation of
    /// // [0,0.8414...,0.9092...]
    /// ```
    ///
    fn sin(self: @Tensor<T>) -> Tensor<T>;
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn cos_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.cos();
    /// }
    /// >>> [8388608,4532384,-3490893]
    /// // The fixed point representation of
    /// // [1, 0.5403...,-0.4161]
    /// ```
    ///
    fn cos(self: @Tensor<T>) -> Tensor<T>;
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn cumsum_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     return tensor.cumsum(axis: 2, exclusive: Option::None(()), reverse: Option::None(()));
    /// }
    /// >>> [[[0,1],[2,5]],[[4,9],[6,13]]]
    /// ```
    ///
    /// Case 2: cumsum with exclusive = true
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn cumsum_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     return tensor.cumsum(axis: 2, exclusive: Option::Some(true), reverse: Option::None(()));
    /// }
    /// >>> [[[0,0],[0,2]],[[0,4],[0,6]]]
    /// ```
    ///
    /// Case 3: cumsum with exclusive = true and reverse = true
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn cumsum_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     return tensor.cumsum(axis: 2, exclusive: Option::Some(true), reverse: Option::Some(true));
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn flatten_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn flatten_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
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
    ///     fn sinh(self: @Tensor<T>) -> Tensor<T>;
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
    /// Returns a new tensor in `T` with the hyperbolic sine of the elements of the input tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn sinh_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.sinh();
    /// }
    /// >>> [[0,9858303],[30424311,84036026]]
    /// // The fixed point representation of
    /// // [[0, 1.175201],[3.62686, 10.0178749]]
    /// ```
    ///
    fn sinh(self: @Tensor<T>) -> Tensor<T>;
    /// # tensor.tanh
    ///
    /// ```rust 
    ///     fn tanh(self: @Tensor<T>) -> Tensor<T>;
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
    /// Returns a new tensor in `T` with the hyperbolic tangent of the elements of the input tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn tanh_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.tanh();
    /// }
    /// >>> [[0,6388715],[8086850,8347125]]
    /// // The fixed point representation of
    /// // [[0, 0.761594],[0.96403, 0.9951]]
    /// ```
    ///
    fn tanh(self: @Tensor<T>) -> Tensor<T>;
    /// # tensor.cosh
    ///
    /// ```rust 
    ///     fn cosh(self: @Tensor<T>) -> Tensor<T>;
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
    /// Returns a new tensor in `T` with the hyperblic cosine of the elements of the input tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn cosh_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.cosh();
    /// }
    /// >>> [[8388608,12944299],[31559585,84453670]]
    /// // The fixed point representation of
    /// // [[, 1.54308],[3.762196, 10.067662]]
    /// ```
    ///
    fn cosh(self: @Tensor<T>) -> Tensor<T>;
    /// # tensor.asinh
    ///
    /// ```rust 
    ///     fn asinh(self: @Tensor<T>) -> Tensor<T>;
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
    /// Returns a new tensor in `T` with the hyperblic sine of the elements of the input tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn asinh_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.asinh();
    /// }
    /// >>> [[0,7393498],[12110093,15254235]]
    /// // The fixed point representation of
    /// // [[0, 0.8814],[1.44364, 1.8185]]
    /// ```
    ///
    fn asinh(self: @Tensor<T>) -> Tensor<T>;
    /// # tensor.acosh
    ///
    /// ```rust 
    ///     fn acosh(self: @Tensor<T>) -> Tensor<T>;
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
    /// Returns a new tensor in `T` with the hyperblic cosine of the elements of the input tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn acosh_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false),
    ///             FixedTrait::new_unscaled(4, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.acosh();
    /// }
    /// >>> [[0,11047444],[14786996,17309365]]
    /// // The fixed point representation of
    /// // [[0, 1.31696],[1.76275, 2.06344]]
    /// ```
    ///
    fn acosh(self: @Tensor<T>) -> Tensor<T>;
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn atan_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.atan();
    /// }
    /// >>> [0,6588397,9287028]
    /// // The fixed point representation of
    /// // [0,0.7853...,1.1071...]
    /// ```
    ///    
    fn atan(self: @Tensor<T>) -> Tensor<T>;
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn asin_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2].span(),
    ///         data: array![FixedTrait::new_unscaled(0, false), FixedTrait::new_unscaled(1, false),]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.asin();
    /// }
    /// >>> [0, 13176794]
    /// // The fixed point representation of
    /// // [0, 1.5707...]
    /// ```
    ///
    fn asin(self: @Tensor<T>) -> Tensor<T>;
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn or_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn or_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![1, 3].span(), data: array![0, 1, 2].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn xor_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn xor_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![1, 3].span(), data: array![0, 1, 2].span(),
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn acos_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2].span(),
    ///         data: array![FixedTrait::new_unscaled(0, false), FixedTrait::new_unscaled(1, false),]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.acos();
    /// }
    /// >>> [13176794, 0]
    /// // The fixed point representation of
    /// // [1.5707..., 0]
    /// ```
    ///
    fn acos(self: @Tensor<T>) -> Tensor<T>;
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn onehot_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2,2].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///             FixedTrait::new_unscaled(3, false),
    ///         ]
    ///             .span(),
    ///     );    
    /// 
    ///     return tensor.onehot(depth: 3, axis: Option::None(()), values: array![0, 1].span());
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn sqrt_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new_unscaled(0, false),
    ///             FixedTrait::new_unscaled(1, false),
    ///             FixedTrait::new_unscaled(2, false),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.sqrt();
    /// }
    /// >>> [0,8388608,11863169]
    /// // The fixed point representation of
    /// // [0,1,1.4142...]
    /// ```
    ///    
    fn sqrt(self: @Tensor<T>) -> Tensor<T>;
    /// # tensor.concat
    ///
    /// ```rust 
    ///    fn concat(tensors: Span<Tensor<T>>, axis: usize,  ) -> Tensor<T>;
    /// ```
    ///
    /// Concatenate a list of tensors into a single tensor.
    ///
    /// ## Args
    ///
    /// * `tensors`(` Span<Tensor<T>>,`) - Array of the input tensors.
    /// * `axis`(`usize`) -  Axis to concat on.
    ///
    /// ## Panics
    ///
    /// * Panic if tensor length is not greater than 1.
    /// * Panics if dimension is not greater than axis.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` concatenated tensor of the input tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn concat_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    ///     let result = TensorTrait::concat(tensors: array![tensor1, tensor2].span(), axis: 0);
    ///     return result;
    /// }
    /// >>> [[0. 1.]
    ///      [2. 3.],
    ///      [0. 1.]
    ///      [2. 3.]]
    ///
    ///     result.shape
    /// >>> (4, 2)
    ///
    ///    let result = TensorTrait::concat(tensors: array![tensor1, tensor2].span(), axis: 1);
    ///    return result;
    /// }
    /// >>> [[0. 1., 0., 1.]
    ///      [2. 3., 2., 3.]]
    ///
    ///     result.shape
    /// >>> (2, 4 ) 
    /// ```
    ///
    fn concat(tensors: Span<Tensor<T>>, axis: usize,) -> Tensor<T>;
    /// # tensor.quantize_linear
    /// 
    /// ```rust
    /// fn quantize_linear(self: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>) -> Tensor::<Q>;
    /// ```
    /// 
    /// Quantizes a Tensor using linear quantization.
    ///
    /// The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point 
    /// to compute the low precision / quantized tensor. The scale factor and zero point must have same shape, 
    /// and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
    /// The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`. For saturation, it saturates to `[-128, 127]`. 
    /// For (x / y_scale), it's rounding to the nearest even.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `y_scale`(`@Tensor<T>`) - Scale for doing quantization to get `y`.
    /// * `y_zero_point`(`@Tensor<T>`) - Zero point for doing quantization to get `y`.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` with the same shape as the input tensor, containing the quantized values.
    ///
    /// ## Type Constraints
    ///
    /// u32 tensor, not supported.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, I32Tensor};
    /// use orion::numbers::{i8, i32, IntegerTrait};
    /// 
    /// fn quantize_linear_example() -> Tensor<i8> {
    ///     // We instantiate a 1D Tensor here.
    ///     let x = TensorTrait::<i32>::new(
    ///         shape: array![6].span(),
    ///         data: array![
    ///             IntegerTrait::new(0, false),
    ///             IntegerTrait::new(2, false),
    ///             IntegerTrait::new(3, false),
    ///             IntegerTrait::new(1000, false),
    ///             IntegerTrait::new(254, true),
    ///             IntegerTrait::new(1000, true),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     // We instantiate the y_scale here.
    ///     let y_scale = TensorTrait::<i32>::new(
    ///         shape: array![1].span(), data: array![IntegerTrait::new(2, false)].span(),
    ///     );
    /// 
    ///     // We instantiate the y_zero_point here.
    ///     let y_zero_point = TensorTrait::<i32>::new(
    ///         shape: array![1].span(), data: array![IntegerTrait::new(1, false)].span(),
    ///     );
    /// 
    ///     return x.quantize_linear(@y_scale, @y_zero_point);
    /// }
    /// >>> [1, 2, 2, 127, -126, -128]
    /// ```
    ///
    fn quantize_linear(
        self: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>
    ) -> Tensor::<i8>;
    /// # tensor.dequantize_linear
    /// 
    /// ```rust
    /// fn dequantize_linear(self: @Tensor<Q>, x_scale: @Tensor<T>, x_zero_point: @Tensor<T>) -> Tensor::<T>;
    /// ```
    /// 
    /// Dequantizes a Tensor using linear dequantization.
    ///
    /// The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute 
    /// the full precision tensor. The dequantization formula is y = (x - x_zero_point) * x_scale. x_scale and 
    /// x_zero_point must have same shape, and can be either a scalar for per-tensor / per layer quantization, 
    /// or a 1-D tensor for per-axis quantization.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `x_scale`(`@Tensor<T>`) - Scale for input `x`.
    /// * `x_zero_point`(`@Tensor<T>`) - Zero point for input `x`.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` with the same shape as the input tensor, containing the dequantized values.
    ///
    /// ## Type Constraints
    ///
    /// u32 tensor, not supported.
    /// fp8x23wide tensor, not supported.
    /// fp16x16wide tensor, not supported.
    ///
    /// ## Examples
    /// 
    /// ```rust
    ///  use array::{ArrayTrait, SpanTrait};
    ///  
    ///  use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, I32Tensor};
    ///  use orion::numbers::{i8, i32, IntegerTrait};
    ///  
    ///  fn dequantize_linear_example() -> Tensor<i32> {
    ///      // We instantiate a 1D Tensor here.
    ///      let x = TensorTrait::<i8>::new(
    ///          shape: array![4].span(),
    ///          data: array![
    ///              IntegerTrait::new(0, false),
    ///              IntegerTrait::new(3, false),
    ///              IntegerTrait::new(125, false),
    ///              IntegerTrait::new(127, false),
    ///          ]
    ///              .span(),
    ///      );
    ///  
    ///      // We instantiate the x_scale here.
    ///      let x_scale = TensorTrait::<i32>::new(
    ///          shape: array![1].span(), data: array![IntegerTrait::new(2, false)].span(),
    ///      );
    ///  
    ///      // We instantiate the x_zero_point here.
    ///      let x_zero_point = TensorTrait::<i32>::new(
    ///          shape: array![1].span(), data: array![IntegerTrait::new(0, false)].span(),
    ///      );
    ///  
    ///      return x.dequantize_linear(@x_scale, @x_zero_point);
    ///  }
    /// >>> [0, 6, 250, 254]
    /// ```
    ///
    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<T>, x_zero_point: @Tensor<T>
    ) -> Tensor::<T>;
    /// # tensor.qlinear_add
    ///
    /// ```rust
    ///     fn qlinear_add(self: @Tensor<i8>, a_scale: @Tensor<T>, a_zero_point: @Tensor<T>, b: @Tensor<i8>, b_scale: @Tensor<T>, b_zero_point: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>) -> Tensor::<i8>;
    /// ```
    /// 
    /// Performs the sum of quantized Tensors
    ///
    /// It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output. 
    /// The quantization formula is y = saturate((x / y_scale) + y_zero_point).
    /// It perfoms the addition of the two vectors once dequantized, then return the quantization of the result of the addition.
    /// The broadcasting is supported
    /// Scale and zero point must have same shape and the same type. They must be either scalar (per tensor) or N-D tensor (per row for 'a' and per column for 'b'). 
    /// Scalar refers to per tensor quantization whereas N-D refers to per row or per column quantization.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<i8>`) - The first tensor to be additionned (a).
    /// * `a_scale`(`@Tensor<T>`) - Scale for input `a`.
    /// * `a_zero_point`(`@Tensor<T>`) - Zero point for input `a`.
    /// * `b`(`@Tensor<i8>`) - The second tensor to be additionned
    /// * `b_scale`(`@Tensor<T>`) - Scale for input `b`.
    /// * `b_zero_point`(`@Tensor<T>`) - Zero point for input `b`.    
    /// * `y_scale`(`@Tensor<T>`) - Scale for outut.
    /// * `y_zero_point`(`@Tensor<T>`) - Zero point for output.   
    ///
    /// ## Returns
    ///
    /// A new `Tensor<i8>`, containing the quantized result of the addition of the dequantized inputs.
    ///
    /// ## Type Constraints
    ///
    /// u32 tensor, not supported.
    /// fp8x23wide tensor, not supported.
    /// fp16x16wide tensor, not supported.
    ///  
    /// ## Example
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, FP16x16Tensor};
    /// use orion::numbers::{i8, FP16x16, FP16x16Impl, IntegerTrait, FixedTrait};
    /// 
    /// 
    /// fn qlinear_add_example() -> Tensor<i8> {    
    ///     let a = TensorTrait::<
    ///         i8
    ///     >::new(
    ///         shape: array![2, 3].span(),
    ///         data: array![
    ///             IntegerTrait::<i8>::new(6_u8, false),
    ///             IntegerTrait::<i8>::new(6_u8, false),
    ///             IntegerTrait::<i8>::new(6_u8, false),
    ///             IntegerTrait::<i8>::new(11_u8, false),
    ///             IntegerTrait::<i8>::new(11_u8, false),
    ///             IntegerTrait::<i8>::new(11_u8, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     // As the operator supports broadcasting shapes [1, 3] and [2, 3] are compatible
    ///     let b = TensorTrait::<
    ///         i8
    ///     >::new(
    ///         shape: array![1, 3].span(),
    ///         data: array![
    ///             IntegerTrait::<i8>::new(40_u8, false),
    ///             IntegerTrait::<i8>::new(40_u8, false),
    ///             IntegerTrait::<i8>::new(40_u8, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     let a_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),);
    ///     let a_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);
    ///     let b_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(16384, false)].span(),);
    ///     let b_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);
    /// 
    ///     let y_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(655360, false)].span(),);
    ///     let y_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, true)].span(),);
    /// 
    ///     return a
    ///         .qlinear_add(
    ///             @a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point
    ///         );
    /// }        
    /// 
    /// >>> [[1, 1, 1], [2, 2, 2]]
    /// ```
    ///
    fn qlinear_add(
        self: @Tensor<i8>,
        a_scale: @Tensor<T>,
        a_zero_point: @Tensor<T>,
        b: @Tensor<i8>,
        b_scale: @Tensor<T>,
        b_zero_point: @Tensor<T>,
        y_scale: @Tensor<T>,
        y_zero_point: @Tensor<T>
    ) -> Tensor::<i8>;
    /// # tensor.qlinear_mul
    ///
    /// ```rust
    ///     fn qlinear_mul(self: @Tensor<i8>, a_scale: @Tensor<T>, a_zero_point: @Tensor<T>, b: @Tensor<i8>, b_scale: @Tensor<T>, b_zero_point: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>) -> Tensor::<i8>;
    /// ```
    /// 
    /// Performs the element-wise multiplication of quantized Tensors
    ///
    /// It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output. 
    /// The quantization formula is y = saturate((x / y_scale) + y_zero_point).
    /// It perfoms the element-wise multiplication of the two vectors once dequantized, then return the quantization of the result of the multiplication.
    /// The broadcasting is supported
    /// Scale and zero point must have same shape and the same type. They must be either scalar (per tensor) or N-D tensor (per row for 'a' and per column for 'b'). 
    /// Scalar refers to per tensor quantization whereas N-D refers to per row or per column quantization.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<i8>`) - The first tensor to be multiplied (a).
    /// * `a_scale`(`@Tensor<T>`) - Scale for input `a`.
    /// * `a_zero_point`(`@Tensor<T>`) - Zero point for input `a`.
    /// * `b`(`@Tensor<i8>`) - The second tensor to be multiplied
    /// * `b_scale`(`@Tensor<T>`) - Scale for input `b`.
    /// * `b_zero_point`(`@Tensor<T>`) - Zero point for input `b`.    
    /// * `y_scale`(`@Tensor<T>`) - Scale for outut.
    /// * `y_zero_point`(`@Tensor<T>`) - Zero point for output.   
    ///
    /// ## Returns
    ///
    /// A new `Tensor<i8>`, containing the quantized result of the element-wise multiplication of the dequantized inputs.
    ///
    /// ## Type Constraints
    ///
    /// u32 tensor, not supported.
    /// fp8x23wide tensor, not supported.
    /// fp16x16wide tensor, not supported.
    /// 
    /// ## Example 
    ///
    ///
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, FP16x16Tensor};
    /// use orion::numbers::{i8, FP16x16, FP16x16Impl, IntegerTrait, FixedTrait};
    /// 
    /// ```rust 
    /// #[test]
    /// #[available_gas(200000000000)]
    /// fn qlinear_mul_example() -> Tensor<i8>{
    ///     let a = TensorTrait::<
    ///         i8
    ///     >::new(
    ///         shape: array![2, 3].span(),
    ///         data: array![
    ///             IntegerTrait::<i8>::new(21_u8, false),
    ///             IntegerTrait::<i8>::new(21_u8, false),
    ///             IntegerTrait::<i8>::new(21_u8, false),
    ///             IntegerTrait::<i8>::new(41_u8, false),
    ///             IntegerTrait::<i8>::new(41_u8, false),
    ///             IntegerTrait::<i8>::new(41_u8, false)
    ///         ]
    ///             .span(),
    ///     );
    ///     let b = TensorTrait::<
    ///         i8
    ///     >::new(
    ///         shape: array![1, 3].span(),
    ///         data: array![
    ///             IntegerTrait::<i8>::new(4_u8, false),
    ///             IntegerTrait::<i8>::new(8_u8, false),
    ///             IntegerTrait::<i8>::new(12_u8, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     let a_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(
    ///         shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),
    ///     );
    ///     let a_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);
    ///     let b_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(16384, false)].span(),);
    ///     let b_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);
    /// 
    ///     let y_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(
    ///         shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(393216, false)].span(),
    ///     );
    ///     let y_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(
    ///         shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(655360, false)].span(),
    ///     );
    /// 
    ///     return = a
    ///         .qlinear_mul(
    ///             @a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point
    ///         );
    /// 
    /// }
    ///
    /// >>> [[16, 23, 30], [23, 36, 50]]
    /// ```
    fn qlinear_mul(
        self: @Tensor<i8>,
        a_scale: @Tensor<T>,
        a_zero_point: @Tensor<T>,
        b: @Tensor<i8>,
        b_scale: @Tensor<T>,
        b_zero_point: @Tensor<T>,
        y_scale: @Tensor<T>,
        y_zero_point: @Tensor<T>
    ) -> Tensor::<i8>;
    /// # tensor.qlinear_matmul
    /// 
    /// ```rust
    ///     fn qlinear_matmul(self: @Tensor<i8>, a_scale: @Tensor<T>, a_zero_point: @Tensor<T>, b: @Tensor<i8>, b_scale: @Tensor<T>, b_zero_point: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>) -> Tensor::<i8>;
    /// ```
    /// 
    /// Multiplies quantized Tensors
    ///
    /// It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output. 
    /// The quantization formula is y = saturate((x / y_scale) + y_zero_point).
    /// It perfoms the multiplication of the two vectors once dequantized. If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes.
    /// Then return the quantization of the result of the multiplication.
    /// Scale and zero point must have same shape and the same type. They must be either scalar (per tensor) or N-D tensor (per row for 'a' and per column for 'b'). 
    /// Scalar refers to per tensor quantization whereas N-D refers to per row or per column quantization.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<i8>`) - The first tensor to be multiplied (a).
    /// * `a_scale`(`@Tensor<T>`) - Scale for input `a`.
    /// * `a_zero_point`(`@Tensor<T>`) - Zero point for input `a`.
    /// * `b`(`@Tensor<i8>`) - The second tensor to be multiplied
    /// * `b_scale`(`@Tensor<T>`) - Scale for input `b`.
    /// * `b_zero_point`(`@Tensor<T>`) - Zero point for input `b`.    
    /// * `y_scale`(`@Tensor<T>`) - Scale for output.
    /// * `y_zero_point`(`@Tensor<T>`) - Zero point for output.   
    ///
    /// ## Returns
    ///
    /// A new `Tensor<i8>`, containing the quantized result of the multiplication of the dequantized inputs.
    ///
    /// ## Type Constraints
    ///
    /// u32 tensor, not supported.
    /// fp8x23wide tensor, not supported.
    /// fp16x16wide tensor, not supported.
    ///  
    /// ## Example
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, FP16x16Tensor};
    /// use orion::numbers::{i8, FP16x16, FP16x16Impl, IntegerTrait, FixedTrait};
    /// fn qlinear_matmul_example() -> Tensor<i8> {    
    ///     let a = TensorTrait::<
    ///         i8
    ///     >::new(
    ///         shape: array![2, 3].span(),
    ///         data: array![
    ///             IntegerTrait::<i8>::new(3_u8, false),
    ///             IntegerTrait::<i8>::new(4_u8, false),
    ///             IntegerTrait::<i8>::new(5_u8, false),
    ///             IntegerTrait::<i8>::new(2_u8, false),
    ///             IntegerTrait::<i8>::new(4_u8, false),
    ///             IntegerTrait::<i8>::new(3_u8, false)
    ///         ]
    ///             .span(),
    ///     );
    ///     let b = TensorTrait::<
    ///         i8
    ///     >::new(
    ///         shape: array![3, 1].span(),
    ///         data: array![
    ///             IntegerTrait::<i8>::new(4_u8, false),
    ///             IntegerTrait::<i8>::new(8_u8, false),
    ///             IntegerTrait::<i8>::new(4_u8, false)
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     let a_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),);
    ///     let a_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);
    ///     let b_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(16384, false)].span(),);
    ///     let b_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);
    /// 
    ///     let y_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(393216, false)].span(),);
    ///     let y_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(655360, false)].span(),);
    /// 
    ///     return a
    ///         .qlinear_matmul(
    ///             @a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point
    ///         );
    /// }        
    /// >>> [14, 13]
    /// ```
    ///
    fn qlinear_matmul(
        self: @Tensor<i8>,
        a_scale: @Tensor<T>,
        a_zero_point: @Tensor<T>,
        b: @Tensor<i8>,
        b_scale: @Tensor<T>,
        b_zero_point: @Tensor<T>,
        y_scale: @Tensor<T>,
        y_zero_point: @Tensor<T>
    ) -> Tensor::<i8>;
    /// # tensor.qlinear_concat
    ///
    /// ```rust 
    ///   qlinear_concat(tensors: Span<Tensor<i8>>, scales: Span<Tensor<T>>, zero_points: Span<Tensor<T>>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>, axis: usize) -> Tensor::<i8>;
    /// ```
    ///
    /// Concatenate a list of tensors after dequantizing them with their respective scales and zero_points and returns the quantized result.
    ///
    /// ## Args
    ///
    /// * `tensors`(` Span<Tensor<i8>>,`) - Array of the quantized input tensors.
    /// * `scales`(` Span<Tensor<T>>,`) - Array of the scales of the quantized input tensors.
    /// * `zero_points`(` Span<Tensor<T>>,`) - Arrayof the zero_points of the quantized input tensors.
    /// * `y_scale`(`@Tensor<T>`) - Scale for output.
    /// * `y_zero_point`(`@Tensor<T>`) - Zero point for output.   
    /// * `axis`(`usize`) -  Axis to concat on.
    ///
    /// ## Panics
    ///
    /// * Panic if tensor length is not greater than 1.
    /// * Panics if dimension is not greater than axis.
    ///
    /// ## Type Constraints
    ///
    /// u32 tensor, not supported.
    /// fp8x23wide tensor, not supported.
    /// fp16x16wide tensor, not supported.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<i8>` concatenated quantized tensor of the dequantized input tensors.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    ///
    /// use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, FP16x16Tensor};
    /// use orion::numbers::{i8, FP16x16, FP16x16Impl, IntegerTrait, FixedTrait};
    /// 
    /// fn qlinear_concat_example() -> Tensor<i8> {
    ///     let tensor1 = TensorTrait::<
    ///         i8
    ///     >::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             IntegerTrait::<i8>::new(5_u8, false),
    ///             IntegerTrait::<i8>::new(5_u8, false),
    ///             IntegerTrait::<i8>::new(5_u8, false),
    ///             IntegerTrait::<i8>::new(5_u8, false),
    ///         ]
    ///             .span(),
    ///     );
    ///     let tensor2 = TensorTrait::<
    ///         i8
    ///     >::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             IntegerTrait::<i8>::new(1_u8, false),
    ///             IntegerTrait::<i8>::new(1_u8, false),
    ///             IntegerTrait::<i8>::new(1_u8, false),
    ///             IntegerTrait::<i8>::new(1_u8, false),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     let tensors = array![tensor1, tensor2].span();
    /// 
    ///     let tensor1_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),);
    ///     let tensor2_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(262144, false)].span(),);
    /// 
    ///     let scales = array![tensor1_scale, tensor2_scale].span();
    /// 
    ///     let tensor1_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(327680, false)].span(),); 
    ///     let tensor2_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);
    /// 
    ///     let zero_points = array![tensor1_zero_point, tensor2_zero_point].span();
    /// 
    ///     let y_scale = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(262144, false)].span(),);
    /// 
    ///     let y_zero_point = TensorTrait::<
    ///         FP16x16
    ///     >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);
    /// 
    ///     return TensorTrait::qlinear_concat(tensors, scales, zero_points, @y_scale, @y_zero_point, 0);
    /// }
    /// 
    /// >>> [[1, 1, 1, 1], [2, 2, 2, 2]]  
    /// ```
    ///
    fn qlinear_concat(
        tensors: Span<Tensor<i8>>,
        scales: Span<Tensor<T>>,
        zero_points: Span<Tensor<T>>,
        y_scale: @Tensor<T>,
        y_zero_point: @Tensor<T>,
        axis: usize
    ) -> Tensor::<i8>;
    /// # tensor.slice
    ///
    /// ```rust 
    ///    fn slice(self: @Tensor<T>, starts: Span<usize>, ends: Span<usize>, axes: Option<Span<usize>>, steps: Option<Span<usize>>) -> Tensor<usize>;
    /// ```
    ///
    /// Produces a slice of the input tensor along multiple axes.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - Tensor of data to extract slices from.
    /// * `starts`(Span<usize>) - 1-D tensor of starting indices of corresponding axis in `axes`
    /// * `ends`(Span<usize>) - 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`
    /// * `axes`(Option<Span<usize>>) - 1-D tensor of axes that `starts` and `ends` apply to. 
    /// * `steps`(Option<Span<usize>>) - 1-D tensor of slice step of corresponding axis in `axes`.    
    ///
    /// ## Panics
    ///
    /// * Panics if the length of starts is not equal to the length of ends.
    /// * Panics if the length of starts is not equal to the length of axes.
    /// * Panics if the length of starts is not equal to the length of steps.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` slice of the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn slice_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 4].span(), 
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(), 
    ///     );
    /// 
    ///     return tensor.slice(
    ///         starts: array![0, 2].span(), 
    ///         ends: array![2, 4].span(), 
    ///         axis: Option::None(()), 
    ///         steps: Option::Some(array![1, 1].span())
    ///     );
    /// }
    /// >>> [[2 3]
    ///      [6 7]]
    /// ```
    ///
    fn slice(
        self: @Tensor<T>,
        starts: Span<usize>,
        ends: Span<usize>,
        axes: Option<Span<usize>>,
        steps: Option<Span<usize>>
    ) -> Tensor<T>;
    /// # tensor.nonzero
    ///
    /// ```rust 
    ///    fn nonzero(self: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Produces indices of the elements that are non-zero (in row-major order - by dimension).
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - Tensor of data to calculate non-zero indices.  
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<usize>` indices of the elements that are non-zero (in row-major order - by dimension).
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn nonzero_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 4].span(), 
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(), 
    ///     );
    /// 
    ///     return tensor.nonzero();
    /// }
    /// >>> [[0 0 0 1 1 1 1]
    ///      [1 2 3 0 1 2 3]]
    /// ```
    ///
    fn nonzero(self: @Tensor<T>) -> Tensor<usize>;
    /// # tensor.gather
    ///
    /// ```rust 
    ///    fn gather(self: @Tensor<T>, indices: Tensor<T>, axis: Option<usize>) -> Tensor<T>;
    /// ```
    ///
    /// Gather entries of the axis dimension of data.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `indices`(`Tensor<T>`) - Tensor of indices.
    /// * `axis`(`Option<usize>`) - Axis to gather on. Default: axis=0.
    ///
    /// ## Panics
    ///
    /// * Panics if index values are not within bounds [-s, s-1] along axis of size s.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` .
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn gather_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 3].span(), 
    ///         data: array![[ 1, 2, 3],[4, 5, 6]].span(), 
    ///     );
    ///     let indices = TensorTrait::<u32>::new(
    ///         shape: array![1, 1].span(), 
    ///         data: array![1, 0].span(), 
    ///     );
    /// 
    ///     return tensor.gather(
    ///         indices: indices, 
    ///         axis: Option::None(()), 
    ///     );
    /// }
    /// >>> [[4. 5. 6.]
    ///      [1. 2. 3.]]
    /// ```
    ///
    fn gather(self: @Tensor<T>, indices: Tensor<usize>, axis: Option<usize>) -> Tensor<T>;
    /// # tensor.unsqueeze
    ///
    /// ```rust 
    ///    fn unsqueeze(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T>;
    /// ```
    ///
    /// Insert single-dimensional entries to the shape of an input tensor (data). Takes one required input axes -
    /// which contains a list of dimension indices and this operator will insert a dimension of value 1 into the
    /// corresponding index of the output tensor (expanded).
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - Tensor of data to unsquezee.
    /// * `axes`(`Span<usize>`) - List of integers indicating the dimensions to be inserted. 
    ///
    /// ## Panics
    ///
    /// * Panics if the given axes have duplicate elements.
    /// * Panics if one of the given axes is invalid.
    ///
    /// ## Returns 
    ///
    /// Reshaped `Tensor<T>` with same data as input.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn unsqueeze_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 4].span(), 
    ///         data: array![0, 1, 2, 3, 4, 5, 6, 7].span(), 
    ///     );
    /// 
    ///     return tensor.unsqueeze(
    ///         axes: array![0, 3].span(), 
    ///     );
    /// }
    /// >>> [[[[0]
    ///        [1]
    ///        [2]
    ///        [3]]
    ///
    ///       [[4]
    ///        [5]
    ///        [6]
    ///        [7]]]]
    /// ```
    ///
    fn unsqueeze(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T>;
    /// # tensor.squeeze
    ///
    /// ```rust 
    ///    fn squeeze(self: @Tensor<T>, axes: Option<Span<i32>>) -> Tensor<T>;
    /// ```
    ///
    /// Removes dimensions of size 1 from the shape of a tensor.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - Tensor of data to calculate non-zero indices.  
    /// * `axes`(`Option<Span<i32>>`) - List of integers indicating the dimensions to squeeze.  
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` Reshaped tensor with same data as input.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn squeeze_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![1, 2, 1, 2, 1].span(), 
    ///         data: array![1, 1, 1, 1].span(), 
    ///     );
    /// 
    ///     return tensor.squeeze(axes: Option::None(());
    /// }
    /// >>> [[1 1]
    ///      [1 1]]
    /// ```
    ///
    fn squeeze(self: @Tensor<T>, axes: Option<Span<i32>>) -> Tensor<T>;
    /// # tensor.clip
    ///
    /// ```rust 
    ///    fn clip(self: @Tensor<T>, min: T, max: T) -> Tensor<T>;
    /// ```
    ///
    /// Clip operator limits the given input within an interval.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - Input tensor whose elements to be clipped.
    /// * `min`(`Option<T>`) - Minimum value, under which element is replaced by min.
    /// * `max`(`Option<T>`) - Maximum value, above which element is replaced by max.
    ///
    /// ## Returns 
    ///
    /// Output `Tensor<T>` with clipped input elements.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn clip_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 3].span(), 
    ///         data: array![[ 1, 2, 3],[4, 5, 6]].span(), 
    ///     );
    /// 
    ///     return tensor.clip(
    ///         min: Option::None(()), 
    ///         max: Option::Some(3),
    ///     );
    /// }
    /// >>> [[1. 2. 3.]
    ///      [3. 3. 3.]]
    /// ```
    ///
    fn clip(self: @Tensor<T>, min: Option<T>, max: Option<T>) -> Tensor<T>;
    /// # tensor.sign
    ///
    /// ```rust 
    ///    fn sign(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Calculates the sign of the given input tensor element-wise.
    /// If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - Tensor of data to calculates the sign of the given input tensor element-wise.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with The sign of the input tensor computed element-wise.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// 
    /// fn sign_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![11].span(), 
    ///         data: array![-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].span(), 
    ///     );
    /// 
    ///     return tensor.sign();
    /// }
    /// >>> [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1]
    /// ```
    ///
    fn sign(self: @Tensor<T>) -> Tensor<T>;
    /// # tensor.identity
    ///
    /// ```rust 
    ///    fn identity(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Return a Tensor with the same shape and contents as input.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - Input tensor.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` to copy input into.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
    /// 
    /// fn identity_example() -> Tensor<FP16x16> {
    ///     let tensor = TensorTrait::<FP16x16>::new(
    ///         shape: array![2, 2].span(), 
    ///         data: array![1, 2, 3, 4].span(), 
    ///     );
    ///     let t_identity = tensor.identity();
    ///     t_identity
    /// }
    /// >>> [[1 2] [3 4]] // A Tensor with the same shape and contents as input
    /// ```
    ///
    fn identity(self: @Tensor<T>) -> Tensor<T>;
    /// #tensor.and
    ///
    /// ```rust
    ///     fn and(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Computes the logical AND of two tensors element-wise.
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn and_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 0, 1, 2, 0, 1, 2].span(),
    ///     );
    /// 
    ///     return tensor_1.and(@tensor_2);
    /// }
    /// >>> [0,1,1,0,1,1,0,1,1]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn and_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![1, 3].span(), data: array![0, 1, 2].span(),
    ///     );
    /// 
    ///     return tensor_1.and(@tensor_2);
    /// }
    /// >>> [0,1,1,0,1,1,0,1,1]
    /// ```
    ///
    fn and(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// #tensor.where
    ///
    /// ```rust
    ///     fn where(self: @Tensor<T>, x: @Tensor<T>, y: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes a new tensor by selecting values from tensor x (resp. y) at
    /// indices where the condition is 1 (resp. 0).
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The condition tensor
    /// * `x`(`@Tensor<T>`) - The first input tensor
    /// * `y`(`@Tensor<T>`) - The second input tensor
    ///
    /// ## Panics
    ///
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Returns
    ///
    /// Return a new `Tensor<T>` of the same shape as the input with elements 
    /// chosen from x or y depending on the condition.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    ///
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    ///
    /// fn where_example() -> Tensor<u32> {
    ///     let tensor_cond = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(), data: array![0, 1, 0, 1].span(),
    ///     );
    ///
    ///     let tensor_x = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(), data: array![2, 4, 6, 8].span(),
    ///     );
    ///
    ///     let tensor_y = TensorTrait::<u32>::new(
    ///         shape: array![2, 2].span(), data: array![1, 3, 5, 9].span(),
    ///     );
    ///
    ///     return tensor_cond.where(@tensor_1, @tensor_2);
    /// }
    /// >>> [1,4,5,8]
    /// ```
    ///
    fn where(self: @Tensor<T>, x: @Tensor<T>, y: @Tensor<T>) -> Tensor<T>;
    /// #tensor.round
    ///
    /// ```rust
    ///     fn round(self: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the round value of all elements in the input tensor.
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    ///
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with 
    /// the round value of all elements in the input tensor.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
    /// use orion::numbers::{FixedTrait, FP16x16};
    /// 
    /// fn round_example() -> Tensor<FP16x16> {
    ///     let tensor = TensorTrait::<FP16x16>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             FixedTrait::new(190054, false),  // 2.9
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return tensor.round();
    /// }
    /// >>> [3]
    /// ```
    ///
    fn round(self: @Tensor<T>) -> Tensor<T>;
    /// # tensor.scatter
    ///
    /// ```rust 
    ///    fn scatter(self: @Tensor<T>, updates: Tensor<T>, indices: Tensor<usize>,  axis: Option<usize>, reduction: Option<usize>) -> Tensor<T>;
    /// ```
    ///
    /// Produces a copy of input data, and updates value to values specified by updates at specific index positions specified by indices.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `updates`(`Tensor<T>`) - The updates tensor.
    /// * `indices`(`Tensor<T>`) - Tensor of indices.
    /// * `axis`(`Option<usize>`) - Axis to scatter on. Default: axis=0.
    /// * `reduction`(`Option<usize>`) - Reduction operation. Default: reduction='none'.
    ///
    /// ## Panics
    ///
    /// * Panics if index values are not within bounds [-s, s-1] along axis of size s.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` .
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn scatter_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![3, 5].span(), 
    ///         data: array![[ 0, 0, 0, 0, 0],
    ///                      [ 0, 0, 0, 0, 0],
    ///                      [ 0, 0, 0, 0, 0]].span(), 
    ///     );
    ///     let updates = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), 
    ///         data: array![[ 1, 2, 3],
    ///                      [ 4, 5, 6],
    ///                      [ 7, 8, 9]].span(), 
    ///     );
    ///     let indices = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), 
    ///         data: array![[ 0, 1, 2],
    ///                      [ 2, 0, 1],
    ///                      [ 1, 0, 1]].span(), 
    ///     );
    /// 
    ///     return tensor.scatter(
    ///         updates: updates
    ///         indices: indices, 
    ///         axis: Option::None(()), 
    ///         reduction: Option::None(()), 
    ///     );
    /// }
    /// >>> [[ 1, 8, 0, 0, 0],
    ///      [ 7, 2, 9, 0, 0],
    ///      [ 4, 0, 3, 0, 0]]
    /// ```
    ///
    fn scatter(
        self: @Tensor<T>,
        updates: Tensor<T>,
        indices: Tensor<usize>,
        axis: Option<usize>,
        reduction: Option<usize>
    ) -> Tensor<T>;
    /// # tensor.trilu
    ///
    /// ```rust 
    ///    fn trilu(self: @Tensor<T>, upper: bool, k: i64) -> Tensor<T>;
    /// ```
    ///
    /// Returns a new tensor with the uppper/lower triangular part of the tensor.
    ///
    /// ## Args
    /// 
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `upper`(`bool`) - if true, returns the upper triangular part of the tensor, otherwise returns the lower part.
    /// * `k`(`i64`) - value corresponding to the number diagonals above or below the main diagonal to exclude or include.
    ///
    /// ## Panics
    ///
    /// * Panics if the dimension of the tensor is less than 2.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` instance with the uppper/lower triangular part of the tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn trilu_tensor_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 3, 3].span(), data: array![0, 4, 3, 2, 0, 9, 8, 2, 5, 2, 7, 2, 2, 6, 0, 2, 6 ,5].span(),
    ///     );
    /// 
    ///     // We can call `trilu` function as follows.
    ///     return tensor.trilu(false, 0);
    /// }
    /// >>> [[[0, 0, 0],[2, 0, 0], [8, 2, 5]], [[2, 0, 0], [2, 6, 0], [2, 6, 5]]]
    /// ```
    ///
    fn trilu(self: @Tensor<T>, upper: bool, k: i64) -> Tensor<T>;
    /// #tensor.bitwise_and
    ///
    /// ```rust
    ///     fn bitwise_and(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
    /// ```
    ///
    /// Computes the bitwise AND of two tensors element-wise.
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
    /// A new `Tensor<T>` with the same shape as the broadcasted inputs.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    ///
    /// fn and_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 0, 4, 5, 0, 6, 2].span(),
    ///     );
    /// 
    ///     return tensor_1.bitwise_and(@tensor_2);
    /// }
    /// >>> [0,1,2,0,4,5,0,6,2]
    /// ```
    ///
    fn bitwise_and(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
    /// ## tensor.reduce_l1
    ///
    /// ```rust 
    ///    fn reduce_l1(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// ```
    ///
    /// Computes the L1 norm of the input tensor's elements along the provided axes.
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
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn reduce_l1_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `reduce_l1` function as follows.
    ///     return tensor.reduce_l1(axis: 1, keepdims: false);
    /// }
    /// >>> [[2,4],[10,12]]
    /// ```
    ///
    fn reduce_l1(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// ## tensor.reduce_l2
    ///
    /// ```rust 
    ///    fn reduce_l2(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// ```
    ///
    /// Computes the L2 norm of the input tensor's elements along the provided axes.
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
    /// fn reduce_l2_example() -> Tensor<u32> {
    ///
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(2);
    ///     shape.append(2);
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FixedTrait::new_unscaled(1, false));
    ///     data.append(FixedTrait::new_unscaled(2, false));
    ///     data.append(FixedTrait::new_unscaled(3, false));
    ///     data.append(FixedTrait::new_unscaled(5, false));
    ///     let tensor = TensorTrait::<FP8x23>::new(shape.span(), data.span());
    /// 
    ///     We can call `reduce_l2` function as follows.
    ///     return tensor.reduce_l2(axis: 1, keepdims: true);
    /// }
    /// >>> [[0x11e3779, 0x2ea5ca1]]
    /// ```
    ///
    fn reduce_l2(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// ## tensor.reduce_sum_square
    ///
    /// ```rust 
    ///    fn reduce_sum_square(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// ```
    ///
    /// Computes the sum square of the input tensor's elements along the provided axes. 
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
    /// fn reduce_sum_square_example() -> Tensor<u32> {
    ///
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(2);
    ///     shape.append(2);
    ///     let mut data = ArrayTrait::new();
    ///     data.append(1);
    ///     data.append(2);
    ///     data.append(3);
    ///     data.append(4);
    ///     let tensor = TensorTrait::<u32>::new(shape.span(), data.span());
    /// 
    ///     We can call `reduce_sum_square` function as follows.
    ///     return tensor.reduce_sum_square(axis: 1, keepdims: true);
    /// }
    /// >>> [[5, 25]]
    /// ```
    ///
    fn reduce_sum_square(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// # tensor.constant_of_shape
    ///
    /// ```rust 
    ///    fn constant_of_shape(shape: Span<usize>, value: T) -> Tensor<T>;
    /// ```
    ///
    /// Returns a new tensor with the given shape and constant value.
    /// 
    /// ## Args
    /// 
    /// * `shape`(`Span<usize>`) - A span representing the shape of the tensor.
    /// * `value` (`T`) - the constant value.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` instance.
    ///
    /// ## Examples
    /// 
    /// Let's create new u32 Tensor with constant 42.
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{
    ///     TensorTrait, // we import the trait
    ///     Tensor, // we import the type
    ///     U32Tensor // we import the implementation. 
    /// };
    /// 
    /// fn constant_of_shape_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::constant_of_shape(shape: array![3].span(), value: 42);
    /// 
    ///     return tensor;
    /// }
    /// 
    /// >>> [42, 42, 42]
    /// ```
    ///
    fn constant_of_shape(shape: Span<usize>, value: T) -> Tensor<T>;
    /// # tensor.binarizer
    /// 
    /// ```rust
    ///  fn binarizer(self: @Tensor<T>, threshold: Option<T>) -> Tensor<T>
    /// ```
    ///
    /// Maps the values of a tensor element-wise to 0 or 1 based on the comparison against a threshold value.
    ///
    /// ## Args
    /// * `self`(`@Tensor<T>`) - The input tensor to be binarized.
    /// * `threshold`(`Option<T>`) - The threshold for the binarization operation.
    ///
    /// ## Returns
    /// A new `Tensor<T>` of the same shape as the input tensor with binarized values.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point numbers.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn binarizer_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(1, false),
    ///             FixedTrait::new(2, false),
    ///             FixedTrait::new(3, false)
    ///         ]
    ///             .span(),
    ///     );
    ///     let threshold = Option::Some(FixedTrait::new(1, false))
    /// 
    ///     return tensor.binarizer(@tensor, threshold);
    /// }
    /// >>> [0, 0, 8388608, 8388608]
    ///     // The fixed point representation of
    ///     [0, 0, 1, 1]
    /// ```
    ///
    fn binarizer(self: @Tensor<T>, threshold: Option<T>) -> Tensor<T>;
    /// # tensor.array_feature_extractor
    ///
    /// ```rust
    ///     fn array_feature_extractor(self: @Tensor<T>, indices: Tensor<usize>) -> Tensor<T>;
    /// ```
    ///
    /// Selects elements of the input tensor based on the indices passed applied to the last tensor axis. 
    /// 
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `indices`(`Tensor<usize>`) - Tensor of indices.
    ///
    /// ## Panics
    ///
    /// * Panics if indices tensor is not 1-dimensional.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` of the same shape as the input tensor with selected elements based on provided indices.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor, U32Tensor};
    /// use orion::numbers::{i32, IntegerTrait};
    /// 
    /// fn array_feature_extractor_example() -> Tensor<i32> {
    ///     let input_tensor = TensorTrait::new(
    ///         shape: array![3, 4].span(),
    ///         data: array![
    ///             IntegerTrait::new(0, false), IntegerTrait::new(1, false), IntegerTrait::new(2, false), IntegerTrait::new(3, false),
    ///             IntegerTrait::new(4, false), IntegerTrait::new(5, false), IntegerTrait::new(6, false), IntegerTrait::new(7, false),
    ///             IntegerTrait::new(8, false), IntegerTrait::new(9, false), IntegerTrait::new(10, false), IntegerTrait::new(11, false)
    ///         ]
    ///             .span(),
    ///     );
    ///     
    ///     let indices = TensorTrait::<u32>::new(
    ///         shape: array![2].span(), data: array![1, 3].span(),
    ///     );
    /// 
    ///     return tensor.array_feature_extractor(@input_tensor, @indices);
    /// }
    /// >>> [[1, 3]
    ///      [5, 7]
    ///      [9, 11]]
    /// ```
    ///
    fn array_feature_extractor(self: @Tensor<T>, indices: Tensor<usize>) -> Tensor<T>;
    /// ## tensor.reduce_mean
    ///
    /// ```rust 
    ///    fn reduce_mean(self: @Tensor<T>, axes: Option<Span<usize>>, keepdims: Option<bool>, noop_with_empty_axes: Option<bool>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the mean of the input tensor's elements along the provided axes.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `axes`(`Option<Span<usize>>`) - Optional input list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor if 'noop_with_empty_axes' is false, else act as an Identity op when 'noop_with_empty_axes' is true.
    /// * `keepdims`(`Option<bool>`) - Keep the reduced dimension or not, default true means keep reduced dimension.
    /// * `noop_with_empty_axes`(`Option<bool>`) - Defines behavior if 'axes' is empty. Default behavior with 'false' is to reduce all axes. When axes is empty and this attribute is set to true, input tensor will not be reduced,and the output tensor would be equivalent to input tensor.
    ///
    /// ## Panics 
    /// 
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` instance with the specified axes reduced by meaning its elements.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn reduce_mean_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `reduce_mean` function as follows.
    ///     return tensor.reduce_mean(axes: array![1].span(), 
    ///         keepdims: Option::None(()), 
    ///         noop_with_empty_axes:  Option::None(()));
    /// }
    /// >>> [[1,2],[5,6]]
    /// ```
    ///
    fn reduce_mean(
        self: @Tensor<T>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<T>;
    /// # tensor.sequence_empty
    ///
    /// ```rust
    ///    fn sequence_empty() -> Array<Tensor<T>>;
    /// ```
    ///
    /// Returns an empty tensor sequence.
    ///
    /// ## Args
    ///
    /// ## Returns
    ///
    /// An empty `Array<Tensor<T>>` instance.
    ///
    /// ## Examples
    ///
    /// Let's create a new empty sequence.
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    ///
    /// use orion::operators::tensor::{
    ///     TensorTrait, // we import the trait
    ///     Tensor, // we import the type
    ///     U32Tensor // we import the implementation. 
    /// };
    ///
    /// fn sequence_empty_example() -> Array<Tensor<u32>> {
    ///     let sequence = TensorTrait::sequence_empty();
    ///
    ///     return sequence;
    /// }
    ///
    /// >>> []
    /// ```
    ///
    fn sequence_empty() -> Array<Tensor<T>>;
    /// # tensor.shrink
    /// 
    /// ```rust
    ///  fn shrink(self: @Tensor<T>, bias: Option<T>, lambd: Option<T>) -> Tensor<T>
    /// ```
    ///
    /// Shrinks the input tensor element-wise to the output tensor with the same datatype and shape based on the following formula:
    /// If x < -lambd: y = x + bias; If x > lambd: y = x - bias; Otherwise: y = 0.
    ///
    /// ## Args
    /// * `self`(`@Tensor<T>`) - The input tensor to be shrinked.
    /// * `bias`(`Option<T>`) - The bias value added to or subtracted from input tensor values.
    /// * `lambd`(`Option<T>`) - The lambd value defining the shrink condition.
    ///
    /// ## Returns
    /// A new `Tensor<T>` of the same datatype and shape as the input tensor with shrinked values.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point numbers.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::numbers::{FixedTrait, FP8x23};
    /// 
    /// fn shrink_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(2, true),
    ///             FixedTrait::new(1, true),
    ///             FixedTrait::new(1, false),
    ///             FixedTrait::new(2, false)
    ///         ]
    ///             .span(),
    ///     );
    ///     let bias = Option::Some(FixedTrait::new(1, false))
    ///     let lambd = Option::Some(FixedTrait::new(1, false))
    /// 
    ///     return tensor.shrink(tensor, bias, lambd);
    /// }
    /// >>> [-8388608, 0, 0, 8388608]
    ///     // The fixed point representation of
    ///     [-1, 0, 0, 1]
    /// ```
    ///
    fn shrink(self: Tensor<T>, bias: Option<T>, lambd: Option<T>) -> Tensor<T>;
    /// ## tensor.sequence_construct
    ///
    /// ```rust 
    ///    fn sequence_construct(tensors: Array<Tensor<T>>) -> Array<Tensor<T>>;
    /// ```
    ///
    /// Constructs a tensor sequence containing the input tensors.
    ///
    /// ## Args
    ///
    /// * `tensors`(`Array<Tensor<T>>`) - The array of input tensors.
    ///
    /// ## Panics 
    /// 
    /// * Panics if input tensor array is empty.
    ///
    /// ## Returns
    ///
    /// A tensor sequence `Array<Tensor<T>>` containing the input tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    ///
    /// fn sequence_construct_example() -> Array<Tensor<usize>> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    ///     let result = TensorTrait::sequence_construct(tensors: array![tensor1, tensor2]);
    ///     return result;
    /// }
    /// >>> [[0, 1, 2, 3], [4, 5, 6, 7]]
    /// ```
    ///
    fn sequence_construct(tensors: Array<Tensor<T>>) -> Array<Tensor<T>>;
    /// ## tensor.reduce_min
    ///
    /// ```rust 
    ///    fn reduce_min(self: @Tensor<T>, axes: Option<Span<usize>>, keepdims: Option<bool>, noop_with_empty_axes: Option<bool>) -> Tensor<T>;
    /// ```
    ///
    /// Computes the min of the input tensor's elements along the provided axes.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `axes`(`Option<Span<usize>>`) - Optional input list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor if 'noop_with_empty_axes' is false, else act as an Identity op when 'noop_with_empty_axes' is true.
    /// * `keepdims`(`Option<bool>`) - Keep the reduced dimension or not, default true means keep reduced dimension.
    /// * `noop_with_empty_axes`(`Option<bool>`) - Defines behavior if 'axes' is empty. Default behavior with 'false' is to reduce all axes. When axes is empty and this attribute is set to true, input tensor will not be reduced,and the output tensor would be equivalent to input tensor.
    ///
    /// ## Panics 
    /// 
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` instance with the specified axes reduced by minimum of its elements.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn reduce_min_example() -> Tensor<u32> {
    ///     let tensor = TensorTrait::<u32>::new(
    ///         shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    ///     );
    /// 
    ///     // We can call `reduce_mean` function as follows.
    ///     return tensor.reduce_min(axes: array![1].span(), 
    ///         keepdims: Option::None(()), 
    ///         noop_with_empty_axes:  Option::None(()));
    /// }
    /// >>> [[0,1],[4,5]]
    /// ```
    ///
    fn reduce_min(
        self: @Tensor<T>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<T>;
    /// # tensor.sequence_insert
    ///
    /// ```rust 
    ///    fn sequence_insert(self: Array<Tensor<T>>, tensor: @Tensor<T>, position: Option<Tensor<i32>>) -> Array<Tensor<T>>;
    /// ```
    ///
    /// Returns a tensor sequence that inserts 'tensor' into 'self' at 'position'.
    ///
    /// ## Args
    ///
    /// * `self`(`Array<Tensor<T>>`) - input sequence.
    /// * `tensor` (`@Tensor<T>`) - the tensor to insert.
    /// * `position` (`@Tensor<i32>`) - the index for insertion (default: -1).
    ///
    /// ## Returns
    ///
    /// Tensor sequence containing 'tensor' inserted into 'self' at 'position'.
    ///
    /// ## Examples
    ///
    /// Let's insert the tensor [2] into the sequence [[1], [3]] at position 1.
    /// use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor, U32Tensor};
    ///
    /// fn sequence_insert_example() -> Array<Tensor<u32>> {
    ///     // Prepare sequence
    ///     let mut sequence = ArrayTrait::new();
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///
    ///     let mut data = ArrayTrait::new();
    ///     data.append(1);
    ///     sequence.append(TensorTrait::new(shape.span(), data.span()));
    ///     let mut data = ArrayTrait::new();
    ///     data.append(3);
    ///
    ///     sequence.append(TensorTrait::new(shape.span(), data.span()));
    ///
    ///     // Prepare input tensor
    ///     let mut data = ArrayTrait::new();
    ///     data.append(2);
    ///     let tensor = TensorTrait::new(shape.span(), data.span());
    ///
    ///     // Prepare position
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     let mut data = ArrayTrait::<i32>::new();
    ///     data.append(i32 { mag: 1, sign: false });
    ///     let position = TensorTrait::<i32>::new(shape.span(), data.span())
    ///
    ///     let sequence = self.sequence_insert(tensor, Option::Some(position));
    ///
    ///     return sequence;
    /// }
    ///
    /// >>> [[1], [2], [3]]
    /// ```
    ///
    fn sequence_insert(self: Array<Tensor<T>>, tensor: @Tensor<T>, position: Option<Tensor<i32>>) -> Array<Tensor<T>>;
    /// ## tensor.sequence_at
    ///
    /// ```rust 
    ///    fn sequence_at(sequence: Array<Tensor<T>>, position: Tensor<i32>) -> Tensor<T>;
    /// ```
    ///
    /// Outputs the tensor at the specified position in the input sequence.
    ///
    /// ## Args
    ///
    /// * `tensors`(`Array<Tensor<T>>`) - The tensor sequence.
    /// * `position`(`Tensor<i32>`) - The position tensor.
    ///
    /// ## Panics 
    /// 
    /// * Panics if position is not a scalar
    /// * Panics if position is out of bounds [-n, n - 1]
    ///
    /// ## Returns
    ///
    /// The tensor `Tensor<T>` from the sequence at the specified position.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor, I32Tensor};
    /// use orion::numbers::{i32, IntegerTrait};
    ///
    /// fn sequence_at_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    ///     
    ///     let mut sequence = ArrayTrait::new();
    ///     sequence.append(tensor1);
    ///     sequence.append(tensor2);
    ///
    ///     let position = TensorTrait::new(shape: array![].span(), data: array![IntegerTrait::new(1, false)].span());
    ///
    ///     let result = TensorTrait::sequence_at(sequence, position);
    ///     return result;
    /// }
    /// >>> [4, 5, 6, 7]
    /// ```
    ///
    fn sequence_at(sequence: Array<Tensor<T>>, position: Tensor<i32>) -> Tensor<T>;
    /// ## tensor.sequence_erase
    ///
    /// ```rust 
    ///    fn sequence_erase(sequence: Array<Tensor<T>>, position: Option<Tensor<i32>>) -> Array<Tensor<T>>;
    /// ```
    ///
    /// Outputs the tensor sequence with the erased tensor at the specified position.
    ///
    /// ## Args
    ///
    /// * `tensors`(`Array<Tensor<T>>`) - The tensor sequence.
    /// * `position`(`Option<Tensor<i32>>`) - The optional position tensor (by default erases the last tensor).
    ///
    /// ## Panics 
    /// 
    /// * Panics if position is not a scalar
    /// * Panics if position is out of bounds [-n, n - 1]
    ///
    /// ## Returns
    ///
    /// The tensor sequence `Array<Tensor<T>>` with the erased tensor at the specified position.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor, I32Tensor};
    /// use orion::numbers::{i32, IntegerTrait};
    ///
    /// fn sequence_erase_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    ///     let tensor3 = TensorTrait::new(shape: array![2, 2].span(), data: array![8, 9, 10, 11].span());
    ///     
    ///     let mut sequence = ArrayTrait::new();
    ///     sequence.append(tensor1);
    ///     sequence.append(tensor2);
    ///     sequence.append(tensor3);
    ///
    ///     let position = TensorTrait::new(shape: array![].span(), data: array![IntegerTrait::new(1, false)].span());
    ///
    ///     let result = TensorTrait::sequence_erase(sequence, position);
    ///     return result;
    /// }
    /// >>> [[0, 1, 2, 3], [8, 9, 10, 11]]
    /// ```
    ///
    fn sequence_erase(sequence: Array<Tensor<T>>, position: Option<Tensor<i32>>) -> Array<Tensor<T>>;
    /// #tensor.pow
    ///
    /// ```rust
    ///     fn pow(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Pow takes input data (Tensor) and exponent Tensor, and produces one output data (Tensor) where the function f(x) = x^exponent, is applied to the data tensor elementwise.
    /// The input tensors must have either:
    /// * Exactly the same shape
    /// * The same number of dimensions and the length of each dimension is either a common length or 1.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The first tensor, base of the exponent.
    /// * `other`(`@Tensor<T>`) - The second tensor, power of the exponent.
    ///
    /// ## Panics
    ///
    /// * Panics if the shapes are not equal or broadcastable
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` with the same shape as the broadcasted inputs.
    ///
    /// ## Examples
    ///
    /// Case 1: Compare tensors with same shape
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn pow_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 0, 1, 2, 0, 1, 2].span(),
    ///     );
    /// 
    ///     return tensor_1.pow(@tensor_2);
    /// }
    /// >>> [0,1,4,0,4,25,0,7,64]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn pow_example() -> Tensor<usize> {
    ///     let tensor_1 = TensorTrait::<u32>::new(
    ///         shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    ///     );
    /// 
    ///     let tensor_2 = TensorTrait::<u32>::new(
    ///         shape: array![1, 3].span(), data: array![0, 1, 2].span(),
    ///     );
    /// 
    ///     return tensor_1.pow(@tensor_2);
    /// }
    /// >>> [0,1,4,0,4,25,0,7,64]
    /// ```
    ///
    fn pow(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
    /// # tensor.sequence_length
    ///
    /// ```rust
    ///    fn sequence_length(self: Array<Tensor<T>>) -> Tensor<u32>;
    /// ```
    ///
    /// Returns the length of the input sequence.
    ///
    /// ## Args
    ///
    /// * `self`(`Array<Tensor<T>>`) - The input sequence.
    ///
    /// ## Returns
    ///
    /// The length of the sequence as scalar, i.e. a tensor of shape [].
    ///
    /// ## Examples
    ///
    /// Let's create new u32 Tensor with constant 42.
    ///
    /// ```rust
    /// let mut sequence = ArrayTrait::new();
    ///
    /// let mut shape = ArrayTrait::<usize>::new();
    /// shape.append(1);
    /// shape.append(2);
    ///
    /// let mut data = ArrayTrait::new();
    /// data.append(3);
    /// data.append(1);
    ///
    /// sequence.append(TensorTrait::new(shape.span(), data.span()));
    ///
    /// sequence.sequence_length()
    /// >>> [1]
    /// ```
    ///
    fn sequence_length(self: Array<Tensor<T>>) -> Tensor<u32>;
    fn concat_from_sequence(sequence: Array<Tensor<T>>, axis: i32, new_axis: Option<usize>) -> Tensor<T>;
}

/// Cf: TensorTrait::new docstring
fn new_tensor<T>(shape: Span<usize>, data: Span<T>) -> Tensor<T> {
    check_shape::<T>(shape, data);
    Tensor::<T> { shape, data }
}

/// Cf: TensorTrait::constant_of_shape docstring
fn constant_of_shape<T, impl FCopy: Copy<T>, impl FDrop: Drop<T>,>(
    shape: Span<usize>, value: T
) -> Tensor<T> {
    let mut data = ArrayTrait::new();

    let mut length = len_from_shape(shape);

    loop {
        match length.into() {
            0 => { break (); },
            _ => {
                data.append(value.clone());
                length -= 1;
            }
        }
    };

    Tensor::<T> { shape, data: data.span() }
}

/// Cf: TensorTrait::ravel_index docstring
fn ravel_index(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
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
            Option::None(_) => { break; }
        };
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
        match shape.pop_front() {
            Option::Some(i) => {
                stride /= *i;

                let coord = remainder / stride;
                remainder = remainder % stride;

                result.append(coord);
            },
            Option::None(_) => { break; }
        };
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
        match shape.pop_back() {
            Option::Some(i) => {
                temp_result.append(accumulated);
                accumulated *= *i;
            },
            Option::None(_) => { break; }
        };
    };

    let mut temp_result = temp_result.span();
    loop {
        match temp_result.pop_back() {
            Option::Some(val) => { result.append(*val); },
            Option::None(_) => { break; }
        };
    };

    return result.span();
}

/// Cf: TensorTrait::reshape docstring
fn reshape<T>(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T> {
    new_tensor(target_shape, *self.data)
}

/// Cf: TensorTrait::at docstring
fn at_tensor<T>(self: @Tensor<T>, indices: Span<usize>) -> @T {
    assert(indices.len() == (*self.shape).len(), 'indices not match dimensions');
    let data = *self.data;

    return data.at(ravel_index(*self.shape, indices));
}

// Return true if two tensor are equal
fn tensor_eq<T, impl TPartialEq: PartialEq<T>>(mut lhs: Tensor<T>, mut rhs: Tensor<T>,) -> bool {
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

/// Cf: TensorTrait::slice docstring
fn slice<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    self: @Tensor<T>,
    starts: Span<usize>,
    ends: Span<usize>,
    axes: Option<Span<usize>>,
    steps: Option<Span<usize>>
) -> Tensor<T> {
    let axes = match axes {
        Option::Some(axes) => axes,
        Option::None(_) => {
            let mut ret: Array<usize> = ArrayTrait::new();
            let mut i: usize = 0;
            let stop_i = starts.len() - 1;
            loop {
                ret.append(i);
                if i == stop_i {
                    break ();
                }
                i += 1;
            };
            ret.span()
        },
    };
    let steps = match steps {
        Option::Some(steps) => steps,
        Option::None(_) => {
            let mut ret: Array<usize> = ArrayTrait::new();
            let mut i: usize = 0;
            let stop_i = starts.len() - 1;
            loop {
                ret.append(1);
                if i == stop_i {
                    break ();
                }
                i += 1;
            };
            ret.span()
        },
    };
    assert(starts.len() == ends.len(), 'Ends and starts len unequal');
    assert(starts.len() == axes.len(), 'Axes and starts len unequal');
    assert(starts.len() == steps.len(), 'Steps and starts len unequal');

    let mut is_empty: bool = false;
    let mut output_shape: Array<usize> = ArrayTrait::new();
    let mut processed_starts: Array<usize> = ArrayTrait::new();
    let mut processed_ends: Array<usize> = ArrayTrait::new();
    let mut processed_steps: Array<usize> = ArrayTrait::new();

    let mut shape = *self.shape;
    let mut i: usize = 0;
    loop {
        match shape.pop_front() {
            Option::Some(ele) => {
                let (axis_index, is_found) = match axes.index_of(i) {
                    Option::Some(axis_index) => (axis_index, true),
                    Option::None(_) => (0, false),
                };

                let mut processed_params = (0, 0, 0, 0);
                if is_found {
                    let mut start: usize = *ele;
                    let mut end: usize = *ele;

                    if *starts.at(axis_index) < *ele {
                        start = *starts.at(axis_index);
                    }

                    if *ele > *ends.at(axis_index) {
                        end = *ends.at(axis_index);
                    };

                    if start > *ele {
                        start = *ele;
                    };

                    if end > *ele {
                        end = *ele;
                    };

                    if start >= end {
                        is_empty = true;
                    } else {
                        let dim = (end - start + (*steps.at(axis_index) - 1))
                            / *steps.at(axis_index);

                        if dim == 0 {
                            is_empty = true;
                        } else {
                            processed_params = (start, end, *steps.at(axis_index), dim);
                        };
                    };
                } else {
                    processed_params = (0, *ele, 1, *ele);
                }

                let (start, end, step, shape) = processed_params;
                processed_starts.append(start);
                processed_ends.append(end);
                processed_steps.append(step);
                output_shape.append(shape);

                i += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let mut output_data: Array<T> = ArrayTrait::new();

    if is_empty {
        return Tensor::<T> { shape: output_shape.span(), data: output_data.span() };
    }

    let mut data = *self.data;
    let mut j: usize = 0;
    loop {
        match data.pop_front() {
            Option::Some(ele) => {
                let mut indices = unravel_index(j, *self.shape);
                let mut is_included = false;

                let mut shape = *self.shape;
                let mut starts = processed_starts.span();
                let mut ends = processed_ends.span();
                let mut steps = processed_steps.span();
                loop {
                    match shape.pop_front() {
                        Option::Some(item) => {
                            let start = *starts.pop_front().unwrap();
                            let end = *ends.pop_front().unwrap();
                            let step = *steps.pop_front().unwrap();
                            let index = *indices.pop_front().unwrap();

                            if index < start || index >= end {
                                is_included = false;
                                break ();
                            }
                            if (index - start) % step == 0 {
                                is_included = true;
                            } else {
                                is_included = false;
                                break ();
                            }
                        },
                        Option::None(_) => { break; }
                    };
                };

                if is_included {
                    output_data.append(*ele);
                }

                j += 1;
            },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(output_shape.span(), output_data.span());
}

/// Cf: TensorTrait::nonzero docstring
fn nonzero<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TPartialEq: PartialEq<T>,
    impl TDrop: Drop<T>,
    impl TCopy: Copy<T>,
    impl TNumber: NumberTrait<T, MAG>
>(
    self: @Tensor<T>
) -> Tensor<usize> {
    let mut indexes_of_dimensions: Array<usize> = ArrayTrait::new();
    let mut self_data_copy = *self.data;
    let mut j: usize = 0;

    loop {
        match self_data_copy.pop_front() {
            Option::Some(val) => {
                if *val != NumberTrait::zero() {
                    let indices = unravel_index(j, *self.shape);
                    let mut i: usize = 0;

                    let mut self_shape_copy = *self.shape;
                    loop {
                        match self_shape_copy.pop_front() {
                            Option::Some(val) => {
                                indexes_of_dimensions.append(*indices.at(i));
                                i += 1;
                            },
                            Option::None(_) => { break (); }
                        };
                    };
                }
                j += 1;
            },
            Option::None(_) => { break (); }
        };
    };

    let indexes_of_dimensions_span = indexes_of_dimensions.span();
    let mut output_data: Array<usize> = ArrayTrait::new();

    if indexes_of_dimensions_span.len() == 0 {
        return Tensor::<
            usize
        > { shape: array![(*self.shape).len(), 0].span(), data: output_data.span() };
    }

    let stop_k = (indexes_of_dimensions_span.len() / (*self.shape).len()) - 1;
    let mut self_shape_copy = *self.shape;
    let mut i: usize = 0;

    loop {
        match self_shape_copy.pop_front() {
            Option::Some(val) => {
                let mut k: usize = 0;

                loop {
                    output_data.append(*indexes_of_dimensions_span.at((*self.shape).len() * k + i));

                    if k == stop_k {
                        break ();
                    }
                    k += 1;
                };
                i += 1;
            },
            Option::None(_) => { break (); }
        };
    };

    return Tensor::<
        usize
    > { shape: array![(*self.shape).len(), stop_k + 1].span(), data: output_data.span() };
}

/// Cf: TensorTrait::squeeze docstring
fn squeeze<T>(self: @Tensor<T>, axes: Option<Span<i32>>) -> Tensor<T> {
    let target_shape = match axes {
        Option::Some(mut axes) => {
            let mut axis_squeezed = 0;
            let mut shape = *self.shape;
            loop {
                match axes.pop_front() {
                    Option::Some(axis) => {
                        let mut reshape: Array<usize> = ArrayTrait::new();
                        let mut index = 0_usize;
                        let axis = if *axis.sign {
                            assert(*axis.mag <= (*self.shape).len(), 'axis out of accepted range');
                            (*self.shape).len() - *axis.mag
                        } else {
                            assert(*axis.mag < (*self.shape).len(), 'axis out of accepted range');
                            *axis.mag
                        };

                        loop {
                            match shape.pop_front() {
                                Option::Some(shape) => {
                                    let squeezed = if axis >= axis_squeezed {
                                        axis - axis_squeezed
                                    } else {
                                        axis
                                    };
                                    if index == squeezed {
                                        assert(*shape == 1, 'shape entry not equal to one');
                                        axis_squeezed += 1;
                                    } else {
                                        reshape.append(*shape);
                                    }
                                },
                                Option::None(_) => { break; },
                            };
                            index += 1;
                        };
                        shape = reshape.span();
                    },
                    Option::None(_) => { break shape; },
                };
            }
        },
        Option::None(_) => {
            let mut reshape: Array<usize> = ArrayTrait::new();
            let mut shape = *self.shape;
            loop {
                match shape.pop_front() {
                    Option::Some(shape) => { if *shape != 1 {
                        reshape.append(*shape);
                    } },
                    Option::None(_) => { break reshape.span(); },
                };
            }
        },
    };

    return Tensor::<T> { shape: target_shape, data: *self.data };
}
/// Cf: TensorTrait::unsqueeze docstring
fn unsqueeze<T>(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T> {
    let dedupped_array = axes.dedup();
    assert(dedupped_array.len() == axes.len(), 'Duplicated input axes');

    let mut self_shape_copy = *self.shape;
    let mut i: usize = 0;
    let mut added_axes_count: usize = 0;
    let mut output_shape: Array<usize> = ArrayTrait::new();
    loop {
        if axes.contains(i + added_axes_count) {
            output_shape.append(1);
            added_axes_count += 1;
        } else {
            match self_shape_copy.pop_front() {
                Option::Some(val) => {
                    output_shape.append(*val);
                    i += 1;
                },
                Option::None(_) => { break (); }
            };
        };
    };

    let mut j: usize = output_shape.len();
    loop {
        if axes.contains(j) {
            output_shape.append(1);
        } else {
            break ();
        }
        j += 1;
    };
    assert(output_shape.len() == axes.len() + (*self.shape).len(), 'Invalid input axes');

    return Tensor::<T> { shape: output_shape.span(), data: *self.data };
}

/// Cf: TensorTrait::sign docstring
fn sign<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialEq: PartialEq<T>,
    impl TDrop: Drop<T>,
    impl TCopy: Copy<T>,
>(
    self: @Tensor<T>
) -> Tensor<T> {
    let mut sign_data_array: Array<T> = ArrayTrait::new();
    let mut data = *self.data;

    loop {
        match data.pop_front() {
            Option::Some(data) => {
                let sign_data = if *data == NumberTrait::zero() {
                    NumberTrait::zero()
                } else if NumberTrait::is_neg(*data) {
                    NumberTrait::neg_one()
                } else {
                    NumberTrait::one()
                };
                sign_data_array.append(sign_data);
            },
            Option::None(_) => {
                break Tensor::<T> { shape: *self.shape, data: sign_data_array.span() };
            }
        };
    }
}

/// Cf: TensorTrait::clip docstring
fn clip<
    T,
    MAG,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TNumber: NumberTrait<T, MAG>
>(
    self: @Tensor<T>, min: Option<T>, max: Option<T>
) -> Tensor<T> {
    let min = match min {
        Option::Some(min) => min,
        Option::None(_) => { NumberTrait::min_value() },
    };
    let max = match max {
        Option::Some(max) => max,
        Option::None(_) => { NumberTrait::max_value() },
    };

    let mut return_data: Array<T> = ArrayTrait::new();
    let mut self_data_copy = *self.data;

    loop {
        match self_data_copy.pop_front() {
            Option::Some(val) => {
                if *val < min {
                    return_data.append(min);
                } else if *val > max {
                    return_data.append(max);
                } else {
                    return_data.append(*val);
                }
            },
            Option::None(_) => { break (); }
        };
    };

    return Tensor::<T> { shape: *self.shape, data: return_data.span() };
}

/// Cf: TensorTrait::identity docstring
fn identity<T>(self: @Tensor<T>) -> Tensor<T> {
    Tensor::<T> { shape: *self.shape, data: *self.data }
}
