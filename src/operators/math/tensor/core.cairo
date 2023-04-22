use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;
use onnx_cairo::operators::math::tensor::helpers::len_from_shape;
use onnx_cairo::operators::math::tensor::helpers::check_shape;

struct Tensor<T> {
    shape: Span<usize>,
    data: @Array<T>
}

impl TensorCopy<T> of Copy<Tensor<T>>;
impl TensorDrop<T> of Drop<Tensor<T>>;

trait TensorTrait<T> {
    fn new(shape: Span<usize>, data: @Array<T>) -> Tensor<T>;
    fn at(self: @Tensor<T>, indices: Span<usize>) -> T;
    fn min(self: @Tensor<T>) -> T;
    fn max(self: @Tensor<T>) -> T;
    fn stride(self: @Tensor<T>) -> Span<usize>;
    fn ravel_index(self: @Tensor<T>, indices: Span<usize>) -> usize;
    fn unravel_index(self: @Tensor<T>, index: usize) -> Span<usize>;
    fn reshape(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T>;
    // REDUCE OPERATIONS
    fn reduce_sum(self: @Tensor<T>, axis: usize) -> Tensor<T>;
    fn argmax(self: @Tensor<T>, axis: usize) -> Tensor<usize>;
}

// --- RAVEL ---

fn ravel_index(shape: Span<usize>, indices: Span<usize>) -> usize {
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

// --- UNRAVEL ---

fn unravel_index(index: usize, shape: Span<usize>) -> Span<usize> {
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

// --- STRIDE ---

fn stride(shape: Span<usize>) -> Span<usize> {
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

// --- Reshape ---

fn reshape<T>(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T> {
    check_shape(target_shape, *self.data);

    Tensor::<T> { shape: target_shape, data: *self.data }
}
