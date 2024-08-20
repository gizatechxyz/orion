use core::num::traits::Zero;
use core::fmt::Debug;
use orion_dl::{Tensor, MutTensor};
use orion_data_structures::vec::{NullableVec, VecTrait};
use core::ops::AddAssign;
use core::cmp::max;


#[derive(Drop, Copy)]
pub(crate) struct ReduceOpMetadata {
    output_indices: Span<usize>,
    output_size: usize,
}

pub(crate) fn tensor_sum_reduce_1d<T, +Add<T>, +Zero<T>, +Copy<T>, +Drop<T>>(
    mut input: Tensor<T>
) -> Tensor<T> {
    let mut result = Zero::<T>::zero();

    loop {
        match input.data.pop_front() {
            Option::Some(input_value) => { result = result + *input_value; },
            Option::None(_) => { break; }
        };
    };

    let mut result_data = ArrayTrait::new();
    result_data.append(result);

    Tensor { data: result_data.span() }
}

pub(crate) fn tensor_sum_reduce_nd<T, +Add<T>, +Copy<T>, +Drop<T>, +Zero<T>>(
    mut input: Tensor<T>, ref metadata: ReduceOpMetadata
) -> MutTensor<T> {
    let mut result_data: NullableVec<T> = VecTrait::new(metadata.output_size);

    loop {
        match input.data.pop_front() {
            Option::Some(input_value) => {
                match metadata.output_indices.pop_front() {
                    Option::Some(output_index) => {
                        let current_sum = result_data.at(*output_index);
                        result_data.set(*output_index, current_sum + *input_value);
                    },
                    Option::None(_) => {
                        break; // This should never happen if metadata is correct
                    }
                }
            },
            Option::None => { break; },
        }
    };

    MutTensor { data: result_data }
}


pub(crate) fn tensor_max_reduce_1d<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(
    mut input: Tensor<T>
) -> Tensor<T> {
    let mut result: Option<T> = Option::None(());

    loop {
        match input.data.pop_front() {
            Option::Some(input_value) => {
                result = match result {
                    Option::Some(current_max) => Option::Some(max(*input_value, current_max)),
                    Option::None(_) => Option::Some(*input_value),
                };
            },
            Option::None(_) => { break; }
        };
    };

    let mut result_data = ArrayTrait::new();
    result_data.append(result.unwrap());

    Tensor { data: result_data.span() }
}

pub(crate) fn tensor_max_reduce_nd<T, +Copy<T>, +Drop<T>, +PartialOrd<T>, +Zero<T>>(
    mut input: Tensor<T>, ref metadata: ReduceOpMetadata
) -> MutTensor<T> {
    let mut result_data: NullableVec<T> = VecTrait::new(metadata.output_size);

    loop {
        match input.data.pop_front() {
            Option::Some(input_value) => {
                match metadata.output_indices.pop_front() {
                    Option::Some(output_index) => {
                        let current_max = result_data.at(*output_index);
                        result_data.set(*output_index, max(*input_value, current_max));
                    },
                    Option::None(_) => {
                        break; // This should never happen if metadata is correct
                    }
                }
            },
            Option::None(_) => { break; }
        };
    };

    MutTensor { data: result_data }
}


#[cfg(test)]
mod tests_sum_reduce {
    use super::{
        Tensor, MutTensor, VecTrait, NullableVec, ReduceOpMetadata, tensor_sum_reduce_1d,
        tensor_sum_reduce_nd
    };


    #[test]
    #[available_gas(20000000)]
    fn test_tensor_sum_reduce_1d() {
        // Test case: Reduce sum along axis 0 for a 1D tensor (full reduction)
        let input_data: Array<u32> = array![1, 2, 3, 4, 5];

        let input = Tensor { data: input_data.span() };

        let result = tensor_sum_reduce_1d(input);

        let expected = array![15]; // [1+2+3+4+5]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        assert_eq!(*result.data.at(0), *expected[0], "Incorrect sum");
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_sum_reduce_2d() {
        // Test case: Reduce sum along axis 1 for a 2x3 tensor
        let input_data: Array<u32> = array![1, 2, 3, 4, 5, 6];
        let output_indices: Array<usize> = array![0, 0, 0, 1, 1, 1];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 2,
        };

        let mut result = tensor_sum_reduce_nd(input, ref metadata);

        let expected = array![6, 15]; // [1+2+3, 4+5+6]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        assert_eq!(result.data.at(0), *expected[0], "Incorrect first sum");
        assert_eq!(result.data.at(1), *expected[1], "Incorrect second sum");
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_sum_reduce_3d_axis0() {
        // Test case: Reduce sum along axis 0 for a 2x2x2 tensor
        let input_data: Array<u32> = array![1, 2, 3, 4, 5, 6, 7, 8];
        let output_indices: Array<usize> = array![0, 1, 2, 3, 0, 1, 2, 3];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 4,
        };

        let mut result = tensor_sum_reduce_nd(input, ref metadata);

        let expected = array![6, 8, 10, 12]; // [1+5, 2+6, 3+7, 4+8]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        let mut i = 0;
        loop {
            if i == expected.len() {
                break;
            }
            assert_eq!(result.data.at(i), *expected[i], "Incorrect sum at index");
            i += 1;
        };
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_sum_reduce_3d_axis1() {
        // Test case: Reduce sum along axis 1 for a 2x3x2 tensor
        let input_data: Array<u32> = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let output_indices: Array<usize> = array![0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 4,
        };

        let mut result = tensor_sum_reduce_nd(input, ref metadata);

        let expected = array![9, 12, 27, 30]; // [1+3+5, 2+4+6, 7+9+11, 8+10+12]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        let mut i = 0;
        loop {
            if i == expected.len() {
                break;
            }
            assert_eq!(result.data.at(i), *expected[i], "Incorrect sum at index");
            i += 1;
        };
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_sum_reduce_4d() {
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

        let mut result = tensor_sum_reduce_nd(input, ref metadata);

        let expected = array![9, 12, 27, 30, 45, 48, 63, 66];
        // [1+3+5, 2+4+6, 7+9+11, 8+10+12, 13+15+17, 14+16+18, 19+21+23, 20+22+24]
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        let mut i = 0;
        loop {
            if i == expected.len() {
                break;
            }
            assert_eq!(result.data.at(i), *expected[i], "Incorrect sum at index");
            i += 1;
        };
    }
}

#[cfg(test)]
mod tests_max_reduce {
    use super::{
        Tensor, MutTensor, VecTrait, NullableVec, ReduceOpMetadata, tensor_max_reduce_1d,
        tensor_max_reduce_nd
    };

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_max_reduce_1d() {
        let input_data: Array<u32> = array![1, 5, 3, 4, 2];
        let input = Tensor { data: input_data.span() };

        let result = tensor_max_reduce_1d(input);

        let expected = array![5];
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        assert_eq!(*result.data.at(0), *expected[0], "Incorrect max");
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_max_reduce_2d() {
        let input_data: Array<u32> = array![1, 2, 3, 4, 5, 6];
        let output_indices: Array<usize> = array![0, 0, 0, 1, 1, 1];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 2,
        };

        let mut result = tensor_max_reduce_nd(input, ref metadata);

        let expected = array![3, 6];
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        assert_eq!(result.data.at(0), *expected[0], "Incorrect first max");
        assert_eq!(result.data.at(1), *expected[1], "Incorrect second max");
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_max_reduce_3d_axis0() {
        let input_data: Array<u32> = array![1, 2, 3, 4, 5, 6, 7, 8];
        let output_indices: Array<usize> = array![0, 1, 2, 3, 0, 1, 2, 3];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 4,
        };

        let mut result = tensor_max_reduce_nd(input, ref metadata);

        let expected = array![5, 6, 7, 8];
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        let mut i = 0;
        loop {
            if i == expected.len() {
                break;
            }
            assert_eq!(result.data.at(i), *expected[i], "Incorrect max at index");
            i += 1;
        };
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_max_reduce_3d_axis1() {
        let input_data: Array<u32> = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let output_indices: Array<usize> = array![0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3];

        let input = Tensor { data: input_data.span() };
        let mut metadata = ReduceOpMetadata {
            output_indices: output_indices.span(), output_size: 4,
        };

        let mut result = tensor_max_reduce_nd(input, ref metadata);

        let expected = array![5, 6, 11, 12];
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        let mut i = 0;
        loop {
            if i == expected.len() {
                break;
            }
            assert_eq!(result.data.at(i), *expected[i], "Incorrect max at index");
            i += 1;
        };
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_max_reduce_4d() {
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

        let mut result = tensor_max_reduce_nd(input, ref metadata);

        let expected = array![5, 6, 11, 12, 17, 18, 23, 24];
        assert_eq!(result.data.len(), expected.len(), "Incorrect result length");
        let mut i = 0;
        loop {
            if i == expected.len() {
                break;
            }
            assert_eq!(result.data.at(i), *expected[i], "Incorrect max at index");
            i += 1;
        };
    }
}
