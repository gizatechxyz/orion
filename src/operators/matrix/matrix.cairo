use core::array::ArrayTrait;
use core::option::OptionTrait;
use orion::numbers::NumberTrait;
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl};
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};

struct MutMatrix<T> {
    data: NullableVec<T>,
    rows: usize,
    cols: usize,
}

impl MutMatrixDestruct<T, +Drop<T>> of Destruct<MutMatrix<T>> {
    fn destruct(self: MutMatrix<T>) nopanic {
        self.data.destruct()
    }
}

#[generate_trait]
impl MutMatrixImpl<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +PartialOrd<T>
> of MutMatrixTrait<T> {
    /// Constructor for the Matrix
    fn new(rows: usize, cols: usize) -> MutMatrix<T> {
        MutMatrix { data: NullableVecImpl::new(), rows: rows, cols: cols }
    }

    /// Get the value at (row, col)
    fn get(ref self: MutMatrix<T>, row: usize, col: usize) -> Option<T> {
        if row >= self.rows || col >= self.cols {
            Option::None
        } else {
            self.data.get(row * self.cols + col)
        }
    }

    /// Set the value at (row, col)
    fn set(ref self: MutMatrix<T>, row: usize, col: usize, value: T) {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            self.data.set(index, value)
        }
    }

    /// Returns the shape of the matrix as (rows, cols)
    fn shape(self: MutMatrix<T>) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the reshaped matrix
    fn reshape<+TensorTrait<T>>(ref self: MutMatrix<T>, target_shape: Span<usize>) -> MutMatrix<T> {
        let mut t = self.to_tensor();
        let mut result = t.reshape(target_shape);
        return result.from_tensor();
    }

    /// Returns the transposed matrix
    fn transpose<+TensorTrait<T>>(ref self: MutMatrix<T>, axes: Span<usize>) -> MutMatrix<T> {
        let mut t = self.to_tensor();
        let mut result = t.transpose(axes);
        return result.from_tensor();
    }

    /// Returns the sum
    fn reduce_sum<+TensorTrait<T>>(ref self: MutMatrix<T>, axis: usize, keepdims: bool) -> MutMatrix<T> {
        let mut t = self.to_tensor();
        let mut result = t.reduce_sum(axis, keepdims);
        return result.from_tensor();
    }

    /// Returns the product of two matrices
    fn matmul<+TensorTrait<T>>(ref self: MutMatrix<T>, ref other: MutMatrix<T>) -> MutMatrix<T> {
        let mut t1 = self.to_tensor();
        let mut t2 = other.to_tensor();
        let mut result = t1.matmul(@t2);
        return result.from_tensor();
    }

    /// Transforms a MutMatrix into a Tensor
    fn to_tensor<+TensorTrait<T>>(ref self: MutMatrix<T>) -> Tensor<T> {
        let mut result_shape = ArrayTrait::<u32>::new();
        result_shape.append(self.rows);
        result_shape.append(self.cols);

        let mut result_data = ArrayTrait::<T>::new();

        let mut i = 0;
        loop {
            if i == self.rows {
                break;
            }
            let mut j = 0;
            loop {
                if j == self.cols {
                    break;
                }
                result_data.append(self.get(i, j).unwrap());
                j += 1;
            };
            i += 1;
        };

        let mut result = TensorTrait::new(result_shape.span(), result_data.span());
        return result;
    }

    /// Transforms a Tensor to a MutMatrix
    fn from_tensor<+TensorTrait<T>>(ref self: Tensor<T>) -> MutMatrix<T> {
        let dim = self.shape.len();
        let result_rows = *self.shape.at(0);
        let result_cols = if (dim == 1) {
                1
            }
            else {
                *self.shape.at(1)
            };
        
        let mut result: MutMatrix = MutMatrixTrait::<T>::new(result_rows, result_cols);

        let mut i = 0;
        loop {
            if i == result_rows {
                break;
            }
            let mut j = 0;
            loop {
                if j == result_cols {
                    break;
                }
                let mut val = if (dim == 1) {
                     self.at(array![i].span())
                }
                else {
                     self.at(array![i, j].span())
                };
                result.set(i, j, val);
                j += 1;
            };
            i += 1;
        };
        return result;
    }

    /// Returns the index of the maximum value along the specified axis
    fn argmax(ref self: MutMatrix<T>, axis: usize) -> Span<usize> {
        assert(axis < 2, 'Invalid axis');

        let mut result: Array<usize> = ArrayTrait::new();
        if axis == 0 {
            let mut col: usize = 0;
            loop {
                if col == self.cols {
                    break;
                }

                let mut max_value = self.get(0, col);
                let mut max_value = match max_value {
                    Option::Some => { max_value.unwrap() },
                    Option::None => { NumberTrait::min_value() }
                };
                let mut max_index = 0;

                let mut row: usize = 1;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let mut value = self.get(row, col);
                    let mut value = match value {
                        Option::Some => { value.unwrap() },
                        Option::None => { NumberTrait::min_value() }
                    };
                    if value > max_value {
                        max_value = value;
                        max_index = row;
                    }

                    row += 1;
                };

                result.append(max_index);

                col += 1;
            };

            return result.span();
        }

        let mut row: usize = 0;
        loop {
            if row == self.rows {
                break;
            }

            let mut max_value = self.get(row, 0);
            let mut max_value = match max_value {
                Option::Some => { max_value.unwrap() },
                Option::None => { NumberTrait::min_value() }
            };
            let mut max_index = 0;

            let mut col: usize = 1;
            loop {
                if col == self.cols {
                    break;
                }

                let mut value = self.get(row, col);
                let mut value = match value {
                    Option::Some => { value.unwrap() },
                    Option::None => { NumberTrait::min_value() }
                };
                if value > max_value {
                    max_value = value;
                    max_index = col;
                }

                col += 1;
            };

            result.append(max_index);

            row += 1;
        };

        return result.span();
    }

    /// Apply softmax to the matrix along the specified axis
    fn softmax<+AddEq<T>, +Div<T>>(ref self: MutMatrix<T>, axis: usize) -> MutMatrix<T> {
        assert(axis < 2, 'Invalid axis');

        let mut result = MutMatrixImpl::new(self.rows, self.cols);

        if axis == 0 {
            let mut col: usize = 0;
            loop {
                if col == self.cols {
                    break;
                }

                let mut sum_exp = NumberTrait::zero();
                let mut row: usize = 0;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    sum_exp += value.exp();

                    row += 1;
                };

                row = 0;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    let softmax_value = (value.exp() / sum_exp).into();
                    result.set(row, col, softmax_value);

                    row += 1;
                };

                col += 1;
            };
        } else {
            let mut row: usize = 0;
            loop {
                if row == self.rows {
                    break;
                }

                let mut sum_exp = NumberTrait::zero();
                let mut col: usize = 0;
                loop {
                    if col == self.cols {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    sum_exp += value.exp();

                    col += 1;
                };

                col = 0;
                loop {
                    if col == self.cols {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    let softmax_value = (value.exp() / sum_exp).into();
                    result.set(row, col, softmax_value);

                    col += 1;
                };

                row += 1;
            };
        }

        result
    }

    /// Apply softmax to the matrix along the specified axis, treating zeros as neutral
    fn softmax_zero<+AddEq<T>, +Div<T>, +PartialEq<T>>(
        ref self: MutMatrix<T>, axis: usize
    ) -> MutMatrix<T> {
        assert(axis < 2, 'Invalid axis');

        let mut result = MutMatrixImpl::new(self.rows, self.cols);

        if axis == 0 {
            let mut col: usize = 0;
            loop {
                if col == self.cols {
                    break;
                }

                let mut sum_exp = NumberTrait::zero();
                let mut row: usize = 0;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    if value != NumberTrait::zero() {
                        sum_exp += value.exp();
                    }
                    row += 1;
                };

                row = 0;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    if value != NumberTrait::zero() {
                        let softmax_value = (value.exp() / sum_exp).into();
                        result.set(row, col, softmax_value);
                    } else {
                        result.set(row, col, NumberTrait::zero());
                    }

                    row += 1;
                };

                col += 1;
            };
        } else {
            let mut row: usize = 0;
            loop {
                if row == self.rows {
                    break;
                }

                let mut sum_exp = NumberTrait::zero();
                let mut col: usize = 0;
                loop {
                    if col == self.cols {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    if value != NumberTrait::zero() {
                        sum_exp += value.exp();
                    }
                    col += 1;
                };

                col = 0;
                loop {
                    if col == self.cols {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();

                    if value != NumberTrait::zero() {
                        let softmax_value = (value.exp() / sum_exp).into();
                        result.set(row, col, softmax_value);
                    } else {
                        result.set(row, col, NumberTrait::zero());
                    }

                    col += 1;
                };

                row += 1;
            };
        }

        result
    }

    /// Apply the sigmoid function to each element of the matrix
    fn sigmoid<+Mul<T>, +Add<T>, +Div<T>>(ref self: MutMatrix<T>) -> MutMatrix<T> {
        let mut result = MutMatrixImpl::new(self.rows, self.cols);

        let mut row: usize = 0;
        loop {
            if row == self.rows {
                break;
            }

            let mut col: usize = 0;
            loop {
                if col == self.cols {
                    break;
                }

                let value = self.get(row, col);
                if value.is_some() {
                    let value = NumberTrait::one()
                        / (NumberTrait::one() + (value.unwrap() * NumberTrait::neg_one()).exp());

                    result.set(row, col, value);
                }

                col += 1;
            };

            row += 1;
        };

        result
    }
    
}
