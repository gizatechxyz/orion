#[derive(Drop, Copy)]
struct Tensor<T> {
    data: Span<T>,
}

#[derive(Drop, Copy)]
struct BinaryOpMetadata {
    lhs_indices: Span<usize>,
    rhs_indices: Span<usize>,
    result_len: usize,
}

fn tensor_add<T, +Add<T>, +Copy<T>, +Drop<T>>(
    lhs: Tensor<T>, rhs: Tensor<T>, metadata: BinaryOpMetadata
) -> Tensor<T> {
    let mut result_data = ArrayTrait::new();

    let mut i: usize = 0;
    loop {
        if i == metadata.result_len {
            break;
        }
        let lhs_value = *lhs.data.at(*metadata.lhs_indices.at(i));
        let rhs_value = *rhs.data.at(*metadata.rhs_indices.at(i));
        result_data.append(lhs_value + rhs_value);
        i += 1;
    };

    Tensor { data: result_data.span() }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, BinaryOpMetadata, tensor_add};

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_add() {
        // This would be precomputed
        let lhs_data: Array<i32> = array![1, 2, 3, 4, 5, 6];
        let rhs_data: Array<i32> = array![10, 20, 30];
        let lhs_indices: Array<usize> = array![0, 1, 2, 3, 4, 5];
        let rhs_indices: Array<usize> = array![0, 1, 2, 0, 1, 2];
        let result_len = 6;

        let lhs = Tensor { data: lhs_data.span() };
        let rhs = Tensor { data: rhs_data.span() };
        let info = BinaryOpMetadata {
            lhs_indices: lhs_indices.span(),
            rhs_indices: rhs_indices.span(),
            result_len: result_len,
        };

        let result = tensor_add(lhs, rhs, info);

        let expected = array![11, 22, 33, 14, 25, 36];
        let mut i = 0;
        loop {
            if i == result.data.len() {
                break;
            }
            assert(*result.data.at(i) == *expected[i], 'Incorrect result');
            i += 1;
        };
    }
}
