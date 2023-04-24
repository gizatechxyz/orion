use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;
use onnx_cairo::operators::math::tensor::helpers::len_from_shape;
use onnx_cairo::operators::math::tensor::helpers::check_shape;

/// A generic Tensor struct representing n-dimensional arrays.
///
/// # Struct fields
/// * `shape` - A span containing the shape of the tensor as usize elements.
/// * `data` - A reference Array of data elements of generic type T.
struct Tensor<T> {
    shape: Span<usize>,
    data: @Array<T>
}

impl TensorCopy<T> of Copy<Tensor<T>>;
impl TensorDrop<T> of Drop<Tensor<T>>;

/// A trait defining the primary operations that can be performed on a Tensor.
///
/// # Functions
/// * `new` - Constructs a new Tensor with the given shape and data array.
/// * `at` - Accesses the element at the given multi-dimensional index.
/// * `min` - Returns the minimum value in the tensor.
/// * `max` - Returns the maximum value in the tensor.
/// * `stride` - Computes the stride of each dimension in the tensor.
/// * `ravel_index` - Converts a multi-dimensional index to a one-dimensional index.
/// * `unravel_index` - Converts a one-dimensional index to a multi-dimensional index.
/// * `reshape` - Returns a new tensor with the specified target shape and the same data.
/// * `transpose` - Returns a new tensor with the axes rearranged according to the given array.
/// * `reduce_sum` - Reduces the tensor by summing along the specified axis.
/// * `argmax` - Returns the index of the maximum value along the specified axis.
trait TensorTrait<T> {
    fn new(shape: Span<usize>, data: @Array<T>) -> Tensor<T>;
    fn at(self: @Tensor<T>, indices: Span<usize>) -> T;
    fn min(self: @Tensor<T>) -> T;
    fn max(self: @Tensor<T>) -> T;
    fn stride(self: @Tensor<T>) -> Span<usize>;
    fn ravel_index(self: @Tensor<T>, indices: Span<usize>) -> usize;
    fn unravel_index(self: @Tensor<T>, index: usize) -> Span<usize>;
    fn reshape(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T>;
    fn transpose(self: @Tensor<T>, axes: @Array<usize>) -> Tensor<T>;
    fn reduce_sum(self: @Tensor<T>, axis: usize) -> Tensor<T>;
    fn argmax(self: @Tensor<T>, axis: usize) -> Tensor<usize>;
}

/// Constructs a new tensor with the given shape and data array after checking compatibility.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
/// * `data` - A reference-counted Array of data elements of type T.
///
/// # Panics
/// * Panics if the shape and data length are incompatible.
///
/// # Returns
/// * A new Tensor with the specified shape and data.
fn new_tensor<T>(shape: Span<usize>, data: @Array<T>) -> Tensor<T> {
    check_shape::<T>(shape, data);
    Tensor::<T> { shape, data }
}

/// Converts a multi-dimensional index into a one-dimensional index for a tensor with the given shape.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
/// * `indices` - A span containing the multi-dimensional index as usize elements.
///
/// # Panics
/// * Panics if the indices are out of bounds for the given shape.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize representing the one-dimensional index.
fn ravel_index(shape: Span<usize>, indices: Span<usize>) -> usize {
    assert(shape.len() == indices.len(), 'shape & indices length unequal');

    let mut raveled_index: usize = 0;

    let mut current_dim: usize = 0;
    loop {
        check_gas();

        let mut first_dim_elements = 1;
        let mut n: usize = current_dim + 1;
        loop {
            check_gas();

            if n == shape.len() {
                break ();
            }

            first_dim_elements *= *shape.at(n);
            n += 1;
        };

        let index = *indices.at(current_dim) * first_dim_elements;
        raveled_index += index;

        current_dim += 1;
        if current_dim == shape.len() {
            break ();
        };
    };

    raveled_index
}

/// Converts a one-dimensional index to a multi-dimensional index for a tensor with the given shape.
///
/// # Arguments
/// * `index` - A usize representing the one-dimensional index.
/// * `shape` - A span containing the shape of the tensor as usize elements.
///
/// # Panics
/// * Panics if the index is out of bounds for the given shape.
/// * Panics if shape is empty.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize representing the multi-dimensional index.
fn unravel_index(index: usize, shape: Span<usize>) -> Span<usize> {
    assert(shape.len() > 0, 'shape cannot be empty');

    let mut result = ArrayTrait::new();
    let mut remainder = index;

    let mut current_dim: usize = 0;
    loop {
        check_gas();

        let mut first_dim_elements = 1;
        let mut n: usize = current_dim + 1;
        loop {
            check_gas();

            if n == shape.len() {
                break ();
            }

            first_dim_elements *= *shape.at(n);
            n += 1;
        };

        let coord = remainder / first_dim_elements;
        remainder = remainder % first_dim_elements;

        result.append(coord);

        current_dim += 1;
        if current_dim >= shape.len() {
            break ();
        };
    };

    return result.span();
}

/// Computes the stride of each dimension in the given shape.
///
/// # Arguments
/// * `shape` - A span containing the shape of the tensor as usize elements.
///
/// # Panics
/// * Panics if shape is empty.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A Span of usize containing the stride for each dimension.
fn stride(shape: Span<usize>) -> Span<usize> {
    assert(shape.len() > 0, 'shape cannot be empty');

    let mut result: Array<usize> = ArrayTrait::new();

    let mut accumulated: usize = 1;

    let mut temp_result = ArrayTrait::new();
    let mut n: usize = shape.len() - 1;
    loop {
        check_gas();

        temp_result.append(accumulated);

        if n == 0 {
            break ();
        }
        accumulated *= *shape.at(n);
        n -= 1;
    };

    let mut i: usize = shape.len() - 1;
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

/// Returns a new tensor with the specified target shape and the same data as the input tensor.
///
/// # Arguments
/// * `self` - A reference-counted Tensor of type T.
/// * `target_shape` - A span containing the target shape of the tensor as usize elements.
///
/// # Panics
/// * Panics if the target shape is incompatible with the input tensor's data.
///
/// # Returns
/// * A new Tensor with the specified target shape and the same data.
fn reshape<T>(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T> {
    new_tensor(target_shape, *self.data)
}
