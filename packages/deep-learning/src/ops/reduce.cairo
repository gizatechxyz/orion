use core::num::traits::Zero;
use core::fmt::Debug;
use orion_dl::Tensor;
use core::ops::AddAssign;


#[derive(Drop, Copy)]
pub(crate) struct ReduceOpMetadata {
    output_indices: Span<usize>,
    output_size: usize,
}

pub(crate) fn tensor_reduce_sum<
    T, +Add<T>, +AddAssign<T, T>, +Zero<T>, +Copy<T>, +Drop<T>, +Debug<T>
>(
    mut input: Tensor<T>, ref metadata: ReduceOpMetadata
) -> Tensor<T> {
    let mut result_data = ArrayTrait::new();
    let mut partial_sums = ArrayTrait::new();
    let mut partial_indices = ArrayTrait::new();

    loop {
        match input.data.pop_front() {
            Option::Some(input_value) => {
                match metadata.output_indices.pop_front() {
                    Option::Some(output_index) => {
                        partial_sums.append(*input_value);
                        partial_indices.append(*output_index);
                    },
                    Option::None(_) => {
                        break; // This should never happen if metadata is correct
                    }
                }
            },
            Option::None(_) => { break; }
        };
    };

    // Combine partial sums
    let mut i = 0;
    loop {
        if i == metadata.output_size {
            break;
        }
        let mut current_sum = Zero::<T>::zero();
        let mut partial_sums_span = partial_sums.span();
        let mut partial_indices_span = partial_indices.span();
        loop {
            match partial_indices_span.pop_front() {
                Option::Some(index) => {
                    let sum = partial_sums_span.pop_front().unwrap();
                    if *index == i {
                        current_sum = current_sum + *sum
                    }
                },
                Option::None(_) => { break; }
            }
        };
        result_data.append(current_sum);
        i += 1;
    };

    Tensor { data: result_data.span() }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, ReduceOpMetadata, tensor_reduce_sum};

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_reduce_sum_2d() {
        // Test case: Reduce sum along axis 1 for a 2x3 tensor
        let input_data: Array<u32> = array![1, 2, 3, 4, 5, 6];
        let output_indices: Array<usize> = array![0, 0, 0, 1, 1, 1];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 2,
        };

        let result = tensor_reduce_sum(input, ref metadata);

        let expected = array![6, 15]; // [1+2+3, 4+5+6]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        assert_eq!(*result.data.at(0), *expected[0], "Incorrect first sum");
        assert_eq!(*result.data.at(1), *expected[1], "Incorrect second sum");
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_reduce_sum_3d_axis0() {
        // Test case: Reduce sum along axis 0 for a 2x2x2 tensor
        let input_data: Array<u32> = array![1, 2, 3, 4, 5, 6, 7, 8];
        let output_indices: Array<usize> = array![0, 1, 2, 3, 0, 1, 2, 3];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 4,
        };

        let result = tensor_reduce_sum(input, ref metadata);

        let expected = array![6, 8, 10, 12]; // [1+5, 2+6, 3+7, 4+8]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        let mut i = 0;
        loop {
            if i == expected.len() {
                break;
            }
            assert_eq!(*result.data.at(i), *expected[i], "Incorrect sum at index");
            i += 1;
        };
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_reduce_sum_3d_axis1() {
        // Test case: Reduce sum along axis 1 for a 2x3x2 tensor
        let input_data: Array<u32> = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let output_indices: Array<usize> = array![0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 4,
        };

        let result = tensor_reduce_sum(input, ref metadata);

        let expected = array![9, 12, 27, 30]; // [1+3+5, 2+4+6, 7+9+11, 8+10+12]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        let mut i = 0;
        loop {
            if i == expected.len() {
                break;
            }
            assert_eq!(*result.data.at(i), *expected[i], "Incorrect sum at index");
            i += 1;
        };
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_reduce_sum_1d() {
        // Test case: Reduce sum along axis 0 for a 1D tensor (full reduction)
        let input_data: Array<u32> = array![1, 2, 3, 4, 5];
        let output_indices: Array<usize> = array![0, 0, 0, 0, 0];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 1,
        };

        let result = tensor_reduce_sum(input, ref metadata);

        let expected = array![15]; // [1+2+3+4+5]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        assert_eq!(*result.data.at(0), *expected[0], "Incorrect sum");
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_reduce_sum_4d() {
        // Test case: Reduce sum along axis 2 for a 2x2x3x2 tensor
        let input_data: Array<u32> = array![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ];
        let output_indices: Array<usize> = array![
            0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7
        ];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 8,
        };

        let result = tensor_reduce_sum(input, ref metadata);

        let expected = array![9, 12, 27, 30, 45, 48, 63, 66];
        // [1+3+5, 2+4+6, 7+9+11, 8+10+12, 13+15+17, 14+16+18, 19+21+23, 20+22+24]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        let mut i = 0;
        loop {
            if i == expected.len() {
                break;
            }
            assert_eq!(*result.data.at(i), *expected[i], "Incorrect sum at index");
            i += 1;
        };
    }
}
