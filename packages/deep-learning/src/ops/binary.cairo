use core::num::traits::{Zero, One};

use orion_dl::Tensor;

#[derive(Drop, Copy)]
pub(crate) struct BinaryOpMetadata {
    lhs_indices: Span<usize>,
    rhs_indices: Span<usize>,
}

pub(crate) fn tensor_add<T, +Add<T>, +Copy<T>, +Drop<T>>(
    lhs: Tensor<T>, rhs: Tensor<T>, ref metadata: BinaryOpMetadata
) -> Tensor<T> {
    let mut result_data = ArrayTrait::new();

    loop {
        match metadata.lhs_indices.pop_front() {
            Option::Some(lhs_index) => {
                match metadata.rhs_indices.pop_front() {
                    Option::Some(rhs_index) => {
                        let lhs_value = *lhs.data.at(*lhs_index);
                        let rhs_value = *rhs.data.at(*rhs_index);
                        result_data.append(lhs_value + rhs_value);
                    },
                    Option::None(_) => {
                        break; // This should never happen if metadata is correct
                    }
                }
            },
            Option::None(_) => { break; }
        };
    };

    Tensor { data: result_data.span() }
}

pub(crate) fn tensor_mul<T, +Mul<T>, +Copy<T>, +Drop<T>>(
    lhs: Tensor<T>, rhs: Tensor<T>, ref metadata: BinaryOpMetadata
) -> Tensor<T> {
    let mut result_data = ArrayTrait::new();

    loop {
        match metadata.lhs_indices.pop_front() {
            Option::Some(lhs_index) => {
                match metadata.rhs_indices.pop_front() {
                    Option::Some(rhs_index) => {
                        let lhs_value = *lhs.data.at(*lhs_index);
                        let rhs_value = *rhs.data.at(*rhs_index);
                        result_data.append(lhs_value * rhs_value);
                    },
                    Option::None(_) => {
                        break; // This should never happen if metadata is correct
                    }
                }
            },
            Option::None(_) => { break; }
        };
    };

    Tensor { data: result_data.span() }
}

pub(crate) fn tensor_rem<T, +Rem<T>, +Copy<T>, +Drop<T>>(
    lhs: Tensor<T>, rhs: Tensor<T>, ref metadata: BinaryOpMetadata
) -> Tensor<T> {
    let mut result_data = ArrayTrait::new();

    loop {
        match metadata.lhs_indices.pop_front() {
            Option::Some(lhs_index) => {
                match metadata.rhs_indices.pop_front() {
                    Option::Some(rhs_index) => {
                        let lhs_value = *lhs.data.at(*lhs_index);
                        let rhs_value = *rhs.data.at(*rhs_index);
                        result_data.append(lhs_value % rhs_value);
                    },
                    Option::None(_) => {
                        break; // This should never happen if metadata is correct
                    }
                }
            },
            Option::None(_) => { break; }
        };
    };

    Tensor { data: result_data.span() }
}

pub(crate) fn tensor_lt<T, +PartialOrd<T>, +Zero<T>, +One<T>, +Copy<T>, +Drop<T>>(
    lhs: Tensor<T>, rhs: Tensor<T>, ref metadata: BinaryOpMetadata
) -> Tensor<T> {
    let mut result_data = ArrayTrait::new();

    loop {
        match metadata.lhs_indices.pop_front() {
            Option::Some(lhs_index) => {
                match metadata.rhs_indices.pop_front() {
                    Option::Some(rhs_index) => {
                        let lhs_value = *lhs.data.at(*lhs_index);
                        let rhs_value = *rhs.data.at(*rhs_index);
                        let r = if lhs_value < rhs_value {
                            One::one()
                        } else {
                            Zero::zero()
                        };
                        result_data.append(r);
                    },
                    Option::None(_) => {
                        break; // This should never happen if metadata is correct
                    }
                }
            },
            Option::None(_) => { break; }
        };
    };

    Tensor { data: result_data.span() }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, BinaryOpMetadata, tensor_add, tensor_mul, tensor_rem, tensor_lt};

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_add() {
        // This would be precomputed
        let lhs_data: Array<i32> = array![1, 2, 3, 4, 5, 6];
        let rhs_data: Array<i32> = array![10, 20, 30];
        let lhs_indices: Array<usize> = array![0, 1, 2, 3, 4, 5];
        let rhs_indices: Array<usize> = array![0, 1, 2, 0, 1, 2];

        let lhs = Tensor { data: lhs_data.span() };
        let rhs = Tensor { data: rhs_data.span() };
        let mut info = BinaryOpMetadata {
            lhs_indices: lhs_indices.span(), rhs_indices: rhs_indices.span(),
        };

        let result = tensor_add(lhs, rhs, ref info);

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

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_mul() {
        // This would be precomputed
        let lhs_data: Array<i32> = array![1, 2, 3, 4, 5, 6];
        let rhs_data: Array<i32> = array![10, 20, 30];
        let lhs_indices: Array<usize> = array![0, 1, 2, 3, 4, 5];
        let rhs_indices: Array<usize> = array![0, 1, 2, 0, 1, 2];

        let lhs = Tensor { data: lhs_data.span() };
        let rhs = Tensor { data: rhs_data.span() };
        let mut info = BinaryOpMetadata {
            lhs_indices: lhs_indices.span(), rhs_indices: rhs_indices.span(),
        };

        let result = tensor_mul(lhs, rhs, ref info);

        let expected = array![10, 40, 90, 40, 100, 180];
        let mut i = 0;
        loop {
            if i == result.data.len() {
                break;
            }
            assert(*result.data.at(i) == *expected[i], 'Incorrect result');
            i += 1;
        };
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_rem() {
        // This would be precomputed
        let lhs_data: Array<i32> = array![10, 11, 12, 13, 14, 15];
        let rhs_data: Array<i32> = array![3, 4, 5];
        let lhs_indices: Array<usize> = array![0, 1, 2, 3, 4, 5];
        let rhs_indices: Array<usize> = array![0, 1, 2, 0, 1, 2];

        let lhs = Tensor { data: lhs_data.span() };
        let rhs = Tensor { data: rhs_data.span() };
        let mut info = BinaryOpMetadata {
            lhs_indices: lhs_indices.span(), rhs_indices: rhs_indices.span(),
        };

        let result = tensor_rem(lhs, rhs, ref info);

        let expected = array![1, 3, 2, 1, 2, 0];
        let mut i = 0;
        loop {
            if i == result.data.len() {
                break;
            }
            assert(*result.data.at(i) == *expected[i], 'Incorrect result');
            i += 1;
        };
    }

    #[test]
    #[available_gas(20000000)]
    fn test_tensor_lt() {
        // This would be precomputed
        let lhs_data: Array<i32> = array![1, 2, 3, 4, 5, 6];
        let rhs_data: Array<i32> = array![1, 42, 2];
        let lhs_indices: Array<usize> = array![0, 1, 2, 3, 4, 5];
        let rhs_indices: Array<usize> = array![0, 1, 2, 0, 1, 2];

        let lhs = Tensor { data: lhs_data.span() };
        let rhs = Tensor { data: rhs_data.span() };
        let mut info = BinaryOpMetadata {
            lhs_indices: lhs_indices.span(), rhs_indices: rhs_indices.span(),
        };

        let result = tensor_lt(lhs, rhs, ref info);

        let expected = array![0, 1, 0, 0, 1, 0];
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
