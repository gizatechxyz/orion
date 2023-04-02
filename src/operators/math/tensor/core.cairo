use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;
use onnx_cairo::operators::math::tensor::helpers::len_from_shape;

struct Tensor<T> {
    shape: @Array<usize>,
    data: @Array<T>
}

impl TensorCopy<T> of Copy::<Tensor::<T>>;
impl TensorDrop<T> of Drop::<Tensor::<T>>;

trait TensorTrait<T> {
    fn new(shape: @Array<usize>, data: @Array<T>) -> Tensor<T>;
    fn at(self: @Tensor<T>, indices: @Array<usize>) -> T;
    fn min(self: @Tensor<T>) -> T;
    fn max(self: @Tensor<T>) -> T;
    fn stride(self: @Tensor<T>) -> Array<usize>;
    fn ravel_index(self: @Tensor<T>, indices: @Array<usize>) -> usize;
    fn unravel_index(self: @Tensor<T>, index: usize) -> Array<usize>;
    fn broadcast_index_mapping(self: @Tensor<T>, indices: @Array<usize>) -> usize;
}

// --- RAVEL ---

fn ravel_index(shape: @Array<usize>, indices: @Array<usize>) -> usize {
    __ravel_index(shape, indices, 0_usize)
}


fn __ravel_index(shape: @Array<usize>, indices: @Array<usize>, current_dim: usize) -> usize {
    check_gas();
    if current_dim == shape.len() - 1_usize {
        return *indices.at(current_dim);
    }

    let first_dim_elements = len_from_shape(shape, current_dim + 1_usize);
    let index = *indices.at(current_dim) * first_dim_elements;
    return index + __ravel_index(shape, indices, current_dim + 1_usize);
}

// --- UNRAVEL ---

fn unravel_index(index: usize, shape: @Array<usize>) -> Array<usize> {
    let mut result = ArrayTrait::new();
    __unravel_index(index, shape, ref result, 0_usize);
    return result;
}

fn __unravel_index(
    index: usize, shape: @Array<usize>, ref result: Array<usize>, current_dim: usize
) {
    check_gas();
    if current_dim == shape.len()
        - 1_usize {
            result.append(index % *shape.at(current_dim));
            return ();
        }

    let first_dim_elements = len_from_shape(shape, current_dim + 1_usize);
    let coord = index / first_dim_elements;
    let remainder = index % first_dim_elements;

    result.append(coord);
    __unravel_index(remainder, shape, ref result, current_dim + 1_usize);
}

// --- BROADCAST INDEX MAPPING ---

fn broadcast_index_mapping(shape: @Array<usize>, indices: @Array<usize>) -> usize {
    let mut result = 0_usize;
    __broadcast_index_mapping(shape, indices, ref result, 0_usize);

    return result;
}

fn __broadcast_index_mapping(
    shape: @Array<usize>, indices: @Array<usize>, ref result: usize, n: usize, 
) {
    check_gas();
    if n == shape.len() {
        return ();
    }

    let stride = stride(shape);
    let index = (*indices.at(n) % *shape.at(n)) * *stride.at(n);
    result += index;
    __broadcast_index_mapping(shape, indices, ref result, n + 1_usize)
}

// --- STRIDE ---

fn stride(shape: @Array<usize>) -> Array<usize> {
    let mut result = ArrayTrait::new();
    __stride(shape, ref result, shape.len() - 1_usize, 1_usize);
    return result;
}

fn __stride(shape: @Array<usize>, ref result: Array<usize>, n: usize, accumulated: usize) {
    check_gas();

    if n == 0_usize {
        result.append(accumulated);
        return ();
    }

    let new_accumulated = accumulated * *shape.at(n);
    __stride(shape, ref result, n - 1_usize, new_accumulated);
    result.append(accumulated);
}
