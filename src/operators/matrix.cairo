use core::array::ArrayTrait;
use core::option::OptionTrait;
use orion::numbers::NumberTrait;
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16};

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

    // fn linalg_solve<+Mul<T>, +Add<T>, +Sub<T>, +Div<T>>(
    //         ref X: MutMatrix<FP16x16>, ref y: MutMatrix<FP16x16>
    //     ) -> MutMatrix<FP16x16> {
        
    //     let n = X.rows;
    //     let mut row: u32 = 0;
    //     let mut col: u32 = 0;
    //     let mut i = 0;


    //     // Forward elimination
    //     loop {
    //         if row == n {
    //             break;
    //         }
            
    //         let mut max_row = row;
    //         i = row + 1;
            
    //         loop {
    //             if i == n {
    //                 break;
    //             }
                
    //             if X.get(i, row).unwrap().mag > X.get(max_row, row).unwrap().mag {
    //                 max_row = i;
    //             }

    //             i += 1;
    //         };

    //         // Get the row max row and set that to the current row
    //         let mut X_row = MutMatrixImpl::new(1, X.cols);
    //         let mut X_max_row = MutMatrixImpl::new(1, X.cols);
    //         let mut y_row = y.get(row, 1).unwrap();
    //         let mut y_max_row = y.get(max_row, 1).unwrap();
            
    //         i = 0;
    //         loop {
    //             if i == n {
    //                 break;
    //             }
                
    //             X_row.set(1, i, X.get(row, i).unwrap());
    //             X_max_row.set(1, i, X.get(max_row, i).unwrap());

    //             i += 1;
    //         };

    //         i = 0;
    //         loop {
    //             if i == n {
    //                 break;
    //             }
                
    //             X.set(row, i, X_max_row.get(1, i).unwrap());
    //             X.set(max_row, i, X_row.get(1, i).unwrap());
                
    //             i += 1;
    //         };

    //         y.set(max_row, 1, y_row);
    //         y.set(row, 1, y_max_row);

    //         i = row + 1;
    //         loop {
    //             if i == n {
    //                 break;
    //             }
    //             let mut factor = X.get(i, row).unwrap() / X.get(row, row).unwrap();
    //             let mut j = row;
    //             loop {
    //                 if j == n {
    //                     break;
    //                 }
    //                 let mut X_new_val = X.get(i, j).unwrap() - factor * X.get(row, j).unwrap();
    //                 X.set(i, j, X_new_val);

    //                 j += 1;
    //             };
    //             let mut y_new_val = y.get(i, 1).unwrap();
    //             y.set(i, 1, y_new_val);

    //             i += 1;
    //         };
            
    //         row += 1;
    //     };

    //     // Back Substitution
    //     let mut S = MutMatrixImpl::new(X.rows, 1);
    //     i = 0;
    //     loop {
    //         if i == n {
    //             break;
    //         }
    //         S.set(i, 1, FP16x16 { mag: 0, sign: true });
    //         i += 1;
    //     };

    //     i = n;
    //     loop {
    //         if i == 0 {
    //             break;
    //         }
    //         let mut X_i = y.get(i - 1, 1).unwrap();
    //         let mut j = i;
    //         loop {
    //             if j == n {
    //                 break;
    //             }
    //             X_i -= X.get(i - 1, j).unwrap() * S.get(j, 1).unwrap();
                
    //             j += 1;
    //         };
    //         X_i /= X.get(i - 1, i - 1).unwrap();
    //         S.set(i - 1, 1, X_i);

    //         i -= 1;
    //     };

    //     return S;
    // }

    
}
