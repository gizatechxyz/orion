#[derive(Drop, Copy)]
struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>
}

impl TensorAdd<T, +Add<T>, +Drop<T>, +Copy<T>> of Add<Tensor<T>> {
    fn add(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<T> {
        // Check if shapes are compatible
        assert(lhs.shape.len() == 2 && rhs.shape.len() == 2, 'Tensors must be 2D');
        assert(lhs.shape[0] == rhs.shape[0] && lhs.shape[1] == rhs.shape[1], 'Shapes must match');

        let mut result_data = ArrayTrait::new();

        // Perform element-wise addition
        let mut i: usize = 0;
        loop {
            if i == lhs.data.len() {
                break;
            }
            let sum = *lhs.data.at(i) + *rhs.data.at(i);
            result_data.append(sum);
            i += 1;
        };

        Tensor { shape: lhs.shape, data: result_data.span() }
    }
}

#[cfg(test)]
mod tests {
    use debug::PrintTrait;
    use super::{Tensor, TensorAdd};

    #[test]
    fn test_tensor_add() {
        // Create two 2x3 tensors
        let shape1 = array![2, 3];
        let data1 = array![1_i32, 2, 3, 4, 5, 6];
        let tensor1 = Tensor { shape: shape1.span(), data: data1.span() };

        let shape2 = array![2, 3];
        let data2 = array![7_i32, 8, 9, 10, 11, 12];
        let tensor2 = Tensor { shape: shape2.span(), data: data2.span() };
        // Perform addition
        let result = TensorAdd::add(tensor1, tensor2);

        // Check shape
        assert(result.shape.len() == 2, 'Result should be 2D');
        assert(*result.shape[0] == 2 && *result.shape[1] == 3, 'Result shape should be 2x3');
        // Check data
        let expected = array![8_i32, 10, 12, 14, 16, 18];
        let mut i: usize = 0;

        loop {
            if i == result.data.len() {
                break;
            }

            assert(*result.data.at(i) == *expected.at(i), 'Incorrect result');

            i += 1;
        };
    }
}
