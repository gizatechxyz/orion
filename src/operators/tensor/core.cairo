use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::utils::check_gas;
use orion::operators::tensor::helpers::{len_from_shape, check_shape};
use orion::numbers::fixed_point::core::FixedType;

struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>
}

impl TensorCopy<T> of Copy<Tensor<T>>;
impl TensorDrop<T> of Drop<Tensor<T>>;

/// Trait
///
/// new - Constructs a new Tensor with the given shape and data array.
/// at - Accesses the element at the given multi-dimensional index.
/// min - Returns the minimum value in the tensor.    
/// max - Returns the maximum value in the tensor.
/// stride - Computes the stride of each dimension in the tensor.
/// ravel_index - Converts a multi-dimensional index to a one-dimensional index.
/// unravel_index - Converts a one-dimensional index to a multi-dimensional index.
/// reshape - Returns a new tensor with the specified target shape and the same data. 
/// transpose - Returns a new tensor with the axes rearranged according to the given array.
/// reduce_sum - Reduces the tensor by summing along the specified axis.
/// argmax - Returns the index of the maximum value along the specified axis.  
/// matmul - Performs matrix multiplication. 
/// exp - Calculates the exponential function (e^x) for each element in a tensor.
/// eq - Check if two tensors are equal element-wise.
trait TensorTrait<T, F> {
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
    /// // 1D TENSOR
    /// fn tensor_1D() -> Tensor<u32> {
    ///     let mut shape = ArrayTrait::new();
    ///     shape.append(3);
    /// 		
    ///     let mut data = ArrayTrait::new();
    ///     data.append(0_u32);
    ///     data.append(1_u32);
    ///     data.append(2_u32);
    /// 		
    ///     let tensor = TensorTrait::<u32>::new(shape.span(), data.span());
    /// 		
    ///     return tensor;
    /// }
    /// 
    /// // 2D TENSOR
    /// fn tensor_2D() -> Tensor<u32> {
    ///     let mut shape = ArrayTrait::new();
    ///     shape.append(2);
    ///     shape.append(2);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(0_u32);
    ///     data.append(1_u32);
    ///     data.append(2_u32);
    ///     data.append(3_u32);
    /// 
    ///     let tensor = TensorTrait::<u32>::new(shape.span(), data.span());
    /// 
    ///     return tensor;
    /// }
    /// 
    /// // 3D TENSOR
    /// fn tensor_3D() -> Tensor<u32> {
    ///     let mut shape = ArrayTrait::new();
    ///     shape.append(2);
    ///     shape.append(2);
    ///     shape.append(2);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(0_u32);
    ///     data.append(1_u32);
    ///     data.append(2_u32);
    ///     data.append(3_u32);
    ///     data.append(4_u32);
    ///     data.append(5_u32);
    ///     data.append(6_u32);
    ///     data.append(7_u32);
    /// 
    ///     let tensor = TensorTrait::<u32>::new(shape.span(), data.span());
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
    /// fn at_example() -> u32 {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
    ///     
    ///     // We set indices to access element of the Tensor.
    ///     let mut indices = ArrayTrait::new();
    ///     indices.append(0);
    ///     indices.append(1);
    ///     indices.append(1);
    /// 		
    ///     // We can call `at` function as follows.
    ///     return tensor.at(indices.span());
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
    /// fn min_example() -> u32 {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
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
    /// fn max_example() -> u32 {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
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
    /// fn stride_example() -> Span<usize> {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
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
    /// fn ravel_index_example() -> usize {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
    ///     
    ///     // We set the indices of the Tensor to ravel.
    ///     let mut indices = ArrayTrait::new();
    ///     indices.append(1);
    ///     indices.append(3);
    ///     indices.append(0);
    /// 		
    ///     // We can call `ravel_index` function as follows.
    ///     return tensor.ravel_index(indices.span());
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
    /// fn unravel_index_example() -> Span<usize> {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
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
    /// fn reshape_tensor_example() -> Tensor<u32> {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
    ///     
    ///     // We set the target shape.
    ///     let mut new_shape = ArrayTrait::new();
    ///     new_shape.append(2);
    ///     new_shape.append(4);
    /// 		
    ///     // We can call `reshape` function as follows.
    ///     return tensor.reshape(new_shape.span());
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
    /// fn transpose_tensor_example() -> Tensor<u32> {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
    /// 
    ///     // We set the axes to be transposed.
    ///     let mut axes = ArrayTrait::new();
    ///     axes.append(1);
    ///     axes.append(2);
    ///     axes.append(0);
    /// 		
    ///     // We can call `transpose` function as follows.
    ///     return tensor.transpose(axes.span());
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
    /// fn reduce_sum_example() -> Tensor<u32> {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
    /// 		
    ///     // We can call `reduce_sum` function as follows.
    ///     return tensor.reduce_sum(0, false);
    /// }
    /// >>> [[4,6],[8,10]]
    /// ```
    ///
    fn reduce_sum(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// # tensor.argmax
    ///
    /// ```rust 
    ///    fn argmax(self: @Tensor<T>, axis: usize) -> Tensor<usize>;
    /// ```
    ///
    /// Returns the index of the maximum value along the specified axis.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `axis`(`usize`) - The axis along which to compute the argmax.
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
    /// ```rust
    /// fn argmax_example() -> Tensor<usize> {
    ///     // We instantiate a 3D Tensor here.
    ///     // [[[0,1],[2,3]],[[4,5],[6,7]]]
    ///     let tensor = u32_tensor_2x2x2_helper();
    /// 		
    ///     // We can call `argmax` function as follows.
    ///     return tensor.argmax(0);
    /// }
    /// >>> [[1,1],[1,1]]
    /// ```
    ///
    fn argmax(self: @Tensor<T>, axis: usize) -> Tensor<usize>;
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
    /// fn dot_product_example() -> Tensor<u32> {
    ///   // We instantiate two 1D Tensor here.
    ///   // tensor_1 = [0,1,2]
    ///   // tensor_2 = [0,1,2]
    ///   let tensor_1 = u32_tensor_1x3_helper();
    ///   let tensor_2 = u32_tensor_1x3_helper();		
    /// 		
    ///   // We can call `matmul` function as follows.
    ///   return tensor_1.matmul(@tensor_2);
    /// }
    /// >>> [5]
    /// ```
    /// 
    /// Case 2: Matrix multiplication (2D \* 2D)
    /// 
    /// ```rust
    /// fn matrix_mul_example() -> Tensor<u32> {
    ///     // We instantiate two 2D Tensor here.
    ///     // tensor_1 = [0,1,2]
    ///     // tensor_2 = [0,1,2]
    ///     let tensor_1 = u32_tensor_2x2_helper();		
    ///     let tensor_2 = u32_tensor_2x2_helper();
    /// 
    ///     // We can call `matmul` function as follows.
    ///     return tensor_1.matmul(@tensor_2);
    /// }
    /// >>> [[2,3],[6,11]]
    /// ```
    /// 
    /// Case 3: Matrix-Vector multiplication (2D x 1D)
    /// 
    /// ```rust
    /// fn matrix_vec_mul_example() -> Tensor<u32> {
    ///     // We instantiate two 2D Tensor here.
    ///     // tensor_1 = [[0,1,2],[3,4,5],[6,7,8]]
    ///     // tensor_2 = [0,1,2]
    ///     let tensor_1 = u32_tensor_3x3_helper();
    ///     let tensor_2 = u32_tensor_1x3_helper();
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
    ///     fn exp(self: @Tensor<T>) -> Tensor<FixedType<F>>;
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
    /// fn exp_example() -> Tensor<FixedType> {
    ///     // We instantiate a 2D Tensor here.
    ///     // [[0,1],[2,3]]
    ///     let tensor = u32_tensor_2x2_helper();
    /// 		
    ///     // We can call `exp` function as follows.
    ///     return tensor.exp();
    /// }
    /// >>> [[8388608,22802594],[61983844,168489688]]
    /// // The fixed point representation of
    /// // [[1, 2.718281],[7.38905, 20.085536]]
    /// ```
    ///
    fn exp(self: @Tensor<T>) -> Tensor<FixedType<F>>;
    /// #tensor.eq
    ///
    /// ```rust
    ///     fn eq(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
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
    /// fn eq_example() -> Tensor<usize> {
    ///     // We instantiate two 3D Tensor here.
    ///     // tensor_y = [[0,1,2],[3,4,5],[6,7,8]]
    ///     // tensor_z = [[0,1,2],[3,4,5],[9,1,5]]
    ///     let tensor_y = u32_tensor_2x2x2_helper();
    ///     let tensor_z = u32_tensor_2x2x2_helper();
    ///     let result = tensor_y.eq(@tensor_z);
    ///     return result;
    /// }
    /// >>> [1,1,1,1,1,0,0,0]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// fn eq_example() -> Tensor<usize> {
    ///     // tensor_y = [[0,1,2],[3,4,5],[6,7,8]]
    ///     // tensor_z = [[0,1,2]]       
    ///     let tensor_y = u32_tensor_3x3_helper();
    ///     let tensor_z = u32_tensor_3x1_helper();
    ///     let result = tensor_y.eq(@tensor_z);
    ///     // We could equally do something like:
    ///     // let result = tensor_z.eq(@tensor_y);
    ///     return result;
    /// }
    /// >>> [1,1,1,0,0,0,0,0,0]
    /// ```
    ///
    fn eq(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
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
    /// fn greater_example() -> Tensor<usize> {
    ///     // We instantiate two 3D Tensor here.
    ///     // tensor_y = [[0,1,2],[3,4,5],[6,7,8]]
    ///     // tensor_z = [[0,1,2],[3,4,5],[9,1,5]]
    ///     let tensor_y = u32_tensor_2x2x2_helper();
    ///     let tensor_z = u32_tensor_2x2x2_helper();
    ///     let result = tensor_y.greater(@tensor_z);
    ///     return result;
    /// }
    /// >>> [0,0,0,0,0,0,0,1,1]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// fn greater_example() -> Tensor<usize> {
    ///     // tensor_y = [[0,1,2],[3,4,5],[6,7,8]]
    ///     // tensor_z = [[0,1,2]]
    ///     let tensor_y = u32_tensor_3x3_helper();
    ///     let tensor_z = u32_tensor_3x1_helper();
    ///     let result = tensor_y.greater(@tensor_z);
    ///     // We could equally do something like:
    ///     // let result = tensor_z.greater(@tensor_y);
    ///     return result;
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
    /// fn greater_equal_example() -> Tensor<usize> {
    ///     // We instantiate two 3D Tensor here.
    ///     // tensor_y = [[0,1,2],[3,4,5],[6,7,8]]
    ///     // tensor_z = [[0,1,2],[3,4,5],[9,1,5]]
    ///     let tensor_y = u32_tensor_2x2x2_helper();
    ///     let tensor_z = u32_tensor_2x2x2_helper();
    ///     let result = tensor_y.greater_equal(@tensor_z);
    ///     return result;
    /// }
    /// >>> [1,1,1,1,1,1,0,1,1]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// fn greater_equal_example() -> Tensor<usize> {
    ///     // tensor_y = [[0,1,2],[3,4,5],[0,0,0]]
    ///     // tensor_z = [[0,1,2]]
    ///     let tensor_y = u32_tensor_3x3_helper();
    ///     let tensor_z = u32_tensor_3x1_helper();
    ///     let result = tensor_y.greater_equal(@tensor_z);
    ///     // We could equally do something like:
    ///     // let result = tensor_z.greater_equal(@tensor_y);
    ///     return result;
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
    /// fn less_example() -> Tensor<usize> {
    ///     // We instantiate two 3D Tensor here.
    ///     // tensor_y = [[0,1,2],[3,4,5],[6,7,8]]
    ///     // tensor_z = [[0,1,2],[3,4,5],[9,1,5]]
    ///     let tensor_y = u32_tensor_2x2x2_helper();
    ///     let tensor_z = u32_tensor_2x2x2_helper();
    ///     let result = tensor_y.less(@tensor_z);
    ///     return result;
    /// }
    /// >>> [0,0,0,0,0,0,1,0,0]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// fn less_example() -> Tensor<usize> {
    ///     // tensor_y = [[0,1,2],[3,4,5],[0,0,0]]
    ///     // tensor_z = [[0,1,2]]       
    ///     let tensor_y = u32_tensor_3x3_helper();
    ///     let tensor_z = u32_tensor_3x1_helper();
    ///     let result = tensor_y.less(@tensor_z);
    ///     // We could equally do something like:
    ///     // let result = tensor_z.less(@tensor_y);
    ///     return result;
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
    /// fn less_equal_example() -> Tensor<usize> {
    ///     // We instantiate two 3D Tensor here.
    ///     // tensor_y = [[0,1,2],[3,4,5],[6,7,8]]
    ///     // tensor_z = [[0,1,2],[3,4,5],[9,1,5]]
    ///     let tensor_y = u32_tensor_2x2x2_helper();
    ///     let tensor_z = u32_tensor_2x2x2_helper();
    ///     let result = tensor_y.less_equal(@tensor_z);
    ///     return result;
    /// }
    /// >>> [1,1,1,1,1,1,1,0,0]
    /// ```
    ///
    /// Case 2: Compare tensors with different shapes
    ///
    /// ```rust
    /// fn less_equal_example() -> Tensor<usize> {
    ///     // tensor_y = [[0,1,2],[3,4,5],[0,0,0]]
    ///     // tensor_z = [[0,1,2]]       
    ///     let tensor_y = u32_tensor_3x3_helper();
    ///     let tensor_z = u32_tensor_3x1_helper();
    ///     let result = tensor_y.less_equal(@tensor_z);
    ///     // We could equally do something like:
    ///     // let result = tensor_z.less_equal(@tensor_y);
    ///     return result;
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
    /// fn abs_example() -> Tensor<i32> {
    ///     // We instantiate a 3D Tensor here.
    ///     // tensor = [[0,-1,2],[-3,4,5],[-6,7,-8]]
    ///     let tensor = i32_tensor_2x2x2_helper();
    ///     let result = tensor.abs();
    ///     return result;
    /// }
    /// >>> [0,1,2,3,4,5,6,7,8]
    /// ```
    ///
    fn abs(self: @Tensor<T>) -> Tensor<T>;
}

/// Cf: TensorTrait::new docstring
fn new_tensor<T>(shape: Span<usize>, data: Span<T>) -> Tensor<T> {
    check_shape::<T>(shape, data);
    Tensor::<T> { shape, data }
}

/// Cf: TensorTrait::ravel_index docstring
fn ravel_index(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    assert(shape.len() == indices.len(), 'shape & indices length unequal');

    let mut raveled_index: usize = 0;
    let mut stride: usize = 1;

    loop {
        check_gas();

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
        check_gas();

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
        check_gas();

        temp_result.append(accumulated);

        if shape.len() == 0 {
            break ();
        }
        accumulated *= *shape.pop_back().unwrap();
    };

    let mut i: usize = shape_len - 1;
    loop {
        check_gas();

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
    new_tensor(target_shape, *self.data)
}

/// Cf: TensorTrait::at docstring
fn at_tensor<T>(self: @Tensor<T>, indices: Span<usize>) -> @T {
    assert(indices.len() == (*self.shape).len(), 'indices not match dimensions');
    let data = *self.data;

    return data.at(ravel_index(*self.shape, indices));
}
