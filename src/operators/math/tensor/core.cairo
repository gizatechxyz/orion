use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;
use onnx_cairo::operators::math::tensor::helpers::len_from_shape;

struct Tensor<T> {
    shape: @Array<usize>,
    data: @Array<T>
}

impl TensorCopy<T> of Copy<Tensor<T>>;
impl TensorDrop<T> of Drop<Tensor<T>>;

trait TensorTrait<T> {
    fn new(shape: @Array<usize>, data: @Array<T>) -> Tensor<T>;
    fn at(self: @Tensor<T>, indices: @Array<usize>) -> T;
    fn min(self: @Tensor<T>) -> T;
    fn max(self: @Tensor<T>) -> T;
    fn stride(self: @Tensor<T>) -> Array<usize>;
    fn ravel_index(self: @Tensor<T>, indices: @Array<usize>) -> usize;
    fn unravel_index(self: @Tensor<T>, index: usize) -> Array<usize>;
    // REDUCE OPERATIONS
    fn reduce_sum(self: @Tensor<T>, axis: usize) -> Tensor<T>;
    fn argmax(self: @Tensor<T>, axis: usize) -> Tensor<usize>;
}

// --- RAVEL ---

fn ravel_index(shape: @Array<usize>, indices: @Array<usize>) -> usize {
    let mut raveled_index: usize = 0;

    let mut i: usize = 0;
    loop {
        check_gas();

        let first_dim_elements = len_from_shape(shape, i + 1);
        let index = *indices.at(i) * first_dim_elements;
        raveled_index += index;

        i += 1;
        if i == shape.len() {
            break ();
        };
    };

    raveled_index
}

// --- UNRAVEL ---

fn unravel_index(index: usize, shape: @Array<usize>) -> Array<usize> {
    let mut result = ArrayTrait::new();
    let mut remainder = index;

    let mut current_dim: usize = 0;
    loop {
        check_gas();

        let first_dim_elements = len_from_shape(shape, current_dim + 1);
        let coord = remainder / first_dim_elements;
        remainder = remainder % first_dim_elements;

        result.append(coord);

        current_dim += 1;
        if current_dim >= shape.len() {
            break ();
        };
    };

    return result;
}

// --- STRIDE ---

fn stride(shape: @Array<usize>) -> Array<usize> {
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

    return result;
}