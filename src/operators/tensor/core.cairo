use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;
use onnx_cairo::operators::tensor::helpers::{len_from_shape, check_shape};
use onnx_cairo::numbers::fixed_point::types::FixedType;

struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>
}

impl TensorCopy<T> of Copy<Tensor<T>>;
impl TensorDrop<T> of Drop<Tensor<T>>;

/// | function                                           | description                                                                 |
/// | -------------------------------------------------- | --------------------------------------------------------------------------- |
/// | [`TensorTrait::new`](tensortrait-new.md)           | Constructs a new Tensor with the given shape and data array.                |
/// | [`tensor.at`](tensor.at.md)                        | Accesses the element at the given multi-dimensional index.                  |
/// | [`tensor.min`](tensor.min.md)                      | Returns the minimum value in the tensor.                                    |
/// | [`tensor.max`](tensor.max.md)                      | Returns the maximum value in the tensor.                                    |
/// | [`tensor.stride`](tensor.stride.md)                | Computes the stride of each dimension in the tensor.                        |
/// | [`tensor.ravel_index`](tensor.ravel\_index.md)     | Converts a multi-dimensional index to a one-dimensional index.              |
/// | [`tensor.unravel_index`](tensor.unravel\_index.md) | Converts a one-dimensional index to a multi-dimensional index.              |
/// | [`tensor.reshape`](tensor.reshape.md)              | Returns a new tensor with the specified target shape and the same data.     |
/// | [`tensor.transpose`](tensor.transpose.md)          | Returns a new tensor with the axes rearranged according to the given array. |
/// | [`tensor.reduce_sum`](tensor.reduce\_sum.md)       | Reduces the tensor by summing along the specified axis.                     |
/// | [`tensor.argmax`](tensor.argmax.md)                | Returns the index of the maximum value along the specified axis.            |
/// | [`tensor.matmul`](tensor.matmul.md)                | Performs matrix multiplication.                                             |
/// | [`tensor.exp`](tensor.exp.md)                      | Calculates the exponential function (e^x) for each element in a tensor.     |
trait TensorTrait<T> {
    /// # TensorTrait::new
    /// 
    /// Returns a new tensor with the given shape and data.
    /// 
    /// ```rust
    /// fn new(shape: Span<usize>, data: Span<T>) -> Tensor<T>;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name    | Type          | Description                                  |
    /// | ------- | ------------- | -------------------------------------------- |
    /// | `shape` | `Span<usize>` | A span representing the shape of the tensor. |
    /// | `data`  | `Span<T>`     | A span containing the array of elements.     |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                                             |
    /// | ----------------------------------------------------- |
    /// | Panics if the shape and data length are incompatible. |
    /// 
    /// #### Returns
    /// 
    /// A new `Tensor<T>` instance.
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Examples
    /// 
    /// Let's create new u32 Tensors.
    /// 
    /// ```rust
    /// /// 1D TENSOR
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
    /// /// 2D TENSOR
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
    /// /// 3D TENSOR
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
    fn new(shape: Span<usize>, data: Span<T>) -> Tensor<T>;
    /// # tensor.argmax
    /// 
    /// Returns the index of the maximum value along the specified axis.
    /// 
    /// ```rust
    /// fn argmax(self: @Tensor<T>, axis: usize) -> Tensor<usize>;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name   | Type         | Description                                 |
    /// | ------ | ------------ | ------------------------------------------- |
    /// | `self` | `@Tensor<T>` | The input tensor.                           |
    /// | `axis` | `usize`      | The axis along which to compute the argmax. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                                                            |
    /// | -------------------------------------------------------------------- |
    /// | Panics if axis is not in the range of the input tensor's dimensions. |
    /// 
    /// #### Returns
    /// 
    /// A new `Tensor<T>` instance containing the indices of the maximum values along the specified axis.
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Examples
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
    fn at(self: @Tensor<T>, indices: Span<usize>) -> T;
    /// # tensor.min
    /// 
    /// Returns the minimum value in the tensor.
    /// 
    /// ```rust
    /// fn min(self: @Tensor<T>) -> T;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name   | Type         | Description       |
    /// | ------ | ------------ | ----------------- |
    /// | `self` | `@Tensor<T>` | The input tensor. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Returns
    /// 
    /// The minimum `T` value in the tensor.
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Examples
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
    fn min(self: @Tensor<T>) -> T;
    /// # tensor.max
    /// 
    /// Returns the maximum value in the tensor.
    /// 
    /// ```rust
    /// fn max(self: @Tensor<T>) -> T;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name   | Type         | Description       |
    /// | ------ | ------------ | ----------------- |
    /// | `self` | `@Tensor<T>` | The input tensor. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Returns
    /// 
    /// The maximum `T` value in the tensor.
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Examples
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
    /// Computes the stride of each dimension in the tensor.
    /// 
    /// ```rust
    /// fn stride(self: @Tensor<T>) -> Span<usize>;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name   | Type         | Description       |
    /// | ------ | ------------ | ----------------- |
    /// | `self` | `@Tensor<T>` | The input tensor. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Returns
    /// 
    /// A span of usize representing the stride for each dimension of the tensor.
    /// 
    /// #### Examples
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
    fn stride(self: @Tensor<T>) -> Span<usize>;
    /// # tensor.ravel\_index
    /// 
    /// Converts a multi-dimensional index to a one-dimensional index.
    /// 
    /// ```rust
    /// fn ravel_index(self: @Tensor<T>, indices: Span<usize>) -> usize;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name      | Type          | Description                         |
    /// | --------- | ------------- | ----------------------------------- |
    /// | `self`    | `@Tensor<T>`  | The input tensor.                   |
    /// | `indices` | `Span<usize>` | The indices of the Tensor to ravel. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                                                    |
    /// | ------------------------------------------------------------ |
    /// | Panics if the indices are out of bounds of the Tensor shape. |
    /// 
    /// #### Returns
    /// 
    /// The index corresponding to the given indices.
    /// 
    /// #### Examples
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
    fn ravel_index(self: @Tensor<T>, indices: Span<usize>) -> usize;
    /// # tensor.unravel\_index
    /// 
    /// Converts a one-dimensional index to a multi-dimensional index.
    /// 
    /// ```rust
    /// fn unravel_index(self: @Tensor<T>, index: usize) -> Span<usize>;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name      | Type          | Description           |
    /// | --------- | ------------- | --------------------- |
    /// | `self`    | `@Tensor<T>`  | The input tensor.     |
    /// | `indices` | `Span<usize>` | The index to unravel. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                                                 |
    /// | --------------------------------------------------------- |
    /// | Panics if the index is out of bounds of the Tensor shape. |
    /// 
    /// #### Returns
    /// 
    /// The unraveled indices corresponding to the given index.
    /// 
    /// #### Examples
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
    fn unravel_index(self: @Tensor<T>, index: usize) -> Span<usize>;
    /// # tensor.reshape
    /// 
    /// Returns a new tensor with the specified target shape and the same data as the input tensor.
    /// 
    /// ```rust
    /// fn reshape(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T>;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name           | Type          | Description                                       |
    /// | -------------- | ------------- | ------------------------------------------------- |
    /// | `self`         | `@Tensor<T>`  | The input tensor.                                 |
    /// | `target_shape` | `Span<usize>` | A span containing the target shape of the tensor. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                                                                |
    /// | ------------------------------------------------------------------------ |
    /// | Panics if the target shape is incompatible with the input tensor's data. |
    /// 
    /// #### Returns
    /// 
    /// A new `Tensor<T>` with the specified target shape and the same data.
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Examples
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
    fn reshape(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T>;
    /// # tensor.transpose
    /// 
    /// Returns a new tensor with the axes rearranged according to the given permutation.
    /// 
    /// ```rust
    /// fn transpose(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T>;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name   | Type          | Description                                                |
    /// | ------ | ------------- | ---------------------------------------------------------- |
    /// | `self` | `@Tensor<T>`  | The input tensor.                                          |
    /// | `axes` | `Span<usize>` | The usize elements representing the axes to be transposed. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                                                                            |
    /// | ------------------------------------------------------------------------------------ |
    /// | Panics if the length of the axes array is not equal to the rank of the input tensor. |
    /// 
    /// #### Returns
    /// 
    /// A `Tensor<T>` instance with the axes reordered according to the given permutation.
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Examples
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
    fn transpose(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T>;
    /// # tensor.reduce\_sum
    /// 
    /// Reduces a tensor by summing its elements along a specified axis.
    /// 
    /// ```rust
    /// fn reduce_sum(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name       | Type         | Description                                        |
    /// | ---------- | ------------ | -------------------------------------------------- |
    /// | `self`     | `@Tensor<T>` | The input tensor.                                  |
    /// | `axis`     | `usize`      | The dimension to reduce.                           |
    /// | `keepdims` | `bool`       | If true, retains reduced dimensions with length 1. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                                                            |
    /// | -------------------------------------------------------------------- |
    /// | Panics if axis is not in the range of the input tensor's dimensions. |
    /// 
    /// #### Returns
    /// 
    /// A new `Tensor<T>` instance with the specified axis reduced by summing its elements.
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Examples
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
    fn reduce_sum(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
    /// # tensor.argmax
    /// 
    /// Returns the index of the maximum value along the specified axis.
    /// 
    /// ```rust
    /// fn argmax(self: @Tensor<T>, axis: usize) -> Tensor<usize>;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name   | Type         | Description                                 |
    /// | ------ | ------------ | ------------------------------------------- |
    /// | `self` | `@Tensor<T>` | The input tensor.                           |
    /// | `axis` | `usize`      | The axis along which to compute the argmax. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                                                            |
    /// | -------------------------------------------------------------------- |
    /// | Panics if axis is not in the range of the input tensor's dimensions. |
    /// 
    /// #### Returns
    /// 
    /// A new `Tensor<T>` instance containing the indices of the maximum values along the specified axis.
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Examples
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
    fn argmax(self: @Tensor<T>, axis: usize) -> Tensor<usize>;
    /// # tensor.matmul
    /// 
    /// Performs matrix product of two tensors.
    /// 
    /// ```rust
    /// fn matmul(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
    /// ```
    /// 
    /// The behavior depends on the dimensionality of the tensors as follows:
    /// 
    /// * If both tensors are 1-dimensional, the dot product is returned.
    /// * If both arguments are 2-dimensional, the matrix-matrix product is returned.
    /// * If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
    /// * If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
    /// 
    /// #### Args
    /// 
    /// | Name    | Type         | Description                        |
    /// | ------- | ------------ | ---------------------------------- |
    /// | `self`  | `@Tensor<T>` | the first tensor to be multiplied  |
    /// | `other` | `@Tensor<T>` | the second tensor to be multiplied |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                                                  |
    /// | ---------------------------------------------------------- |
    /// | Panics if the dimension of the tensors is higher than two. |
    /// 
    /// #### Returns
    /// 
    /// A new `Tensor<T>` resulting from the matrix multiplication.
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Examples
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
    fn matmul(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
    /// # tensor.exp
    /// 
    /// Computes the exponential of all elements of the input tensor.
    /// 
    /// $$
    /// y_i=e^{x_i}
    /// $$
    /// 
    /// ```rust
    /// fn exp(self: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name   | Type         | Description       |
    /// | ------ | ------------ | ----------------- |
    /// | `self` | `@Tensor<T>` | The input tensor. |
    /// 
    /// > _`<T>` generic type depends on Tensor dtype._
    /// 
    /// #### Returns
    /// 
    /// Returns a new tensor in [`FixedType`](../../numbers/fixed-point/) with the exponential of the elements of the input tensor.
    /// 
    /// #### Examples
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
    /// >>> [[67108864,182420802],[495871144,1347917552]]
    /// // The fixed point representation of
    /// // [[1, 2.718281],[7.38905, 20.085536]]
    /// ```
    fn exp(self: @Tensor<T>) -> Tensor<FixedType>;
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
