mod operators;
mod numbers;
mod utils;
mod test_helper;

use core::debug::PrintTrait;
use core::array::ArrayTrait;
use core::option::OptionTrait;
use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::implementations::fp16x16::math::core::{ceil, abs};
use orion::operators::matrix::{MutMatrix, MutMatrixTrait, MutMatrixImpl};
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FixedTrait, FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;

fn test_matrix(ref X: MutMatrix<FP16x16>) {
        // Print X by columns
        let mut c = 0;
        loop {
            if c == X.cols {
                break ();
            }
            let mut r = 0;
            loop {
                if r == X.rows {
                    break;
                }
                let mut val = X.get(r, c).unwrap();
                val.print();
                r += 1;
            };
            c += 1;
        };
    }

fn linalg_solve(ref X: MutMatrix<FP16x16>, ref y: MutMatrix<FP16x16>) -> MutMatrix<FP16x16> {
        
        let n = X.rows;
        let mut row: u32 = 0;
        let mut col: u32 = 0;
        let mut i = 0;

        loop {
            if row == n {
                break;
            }
            
            // Find the row number and max row number for X
            i = row + 1;
            let mut max_row = row;
            loop {
                if i == n {
                    break;
                }
                if X.get(i, row).unwrap().mag > X.get(max_row, row).unwrap().mag {
                    max_row = i;
                }
                i += 1;
            };

            let mut X_row = MutMatrixImpl::new(1, X.cols);
            let mut X_max_row = MutMatrixImpl::new(1, X.cols);
            let mut y_row = y.get(row, 0).unwrap();
            let mut y_max_row = y.get(max_row, 0).unwrap();

            // Store X_row and X_max_row
            i = 0;
            loop {
                if i == n {
                    break;
                }

                X_row.set(0, i, X.get(row, i).unwrap());
                X_max_row.set(0, i, X.get(max_row, i).unwrap());

                i += 1;
            };

            // Interchange X_row with X_max_row, y_row with y_max_row
            i = 0;
            loop {
                if i == n {
                    break;
                }
                
                X.set(row, i, X_max_row.get(0, i).unwrap());
                X.set(max_row, i, X_row.get(0, i).unwrap());
                
                i += 1;
            };
            y.set(max_row, 0, y_row);
            y.set(row, 0, y_max_row);

            // Check for singularity
            assert(X.get(row, row).unwrap().mag != 0, 'Singular matrix error');

            // Perform forward elimination
            i = row + 1;
            loop {
                if i == n {
                    break;
                }
                let mut factor = X.get(i, row).unwrap() / X.get(row, row).unwrap();
                let mut j = row;
                loop {
                    if j == n {
                        break;
                    }
                    let mut X_new_val = X.get(i, j).unwrap() - factor * X.get(row, j).unwrap();
                    X.set(i, j, X_new_val);

                    j += 1;
                };
                let mut y_new_val = y.get(i, 0).unwrap() - factor * y.get(row, 0).unwrap();
                y.set(i, 0, y_new_val);

                i += 1;
            };
            
            row += 1;
        };

        // Perform back substitution
        let mut S = MutMatrixImpl::new(X.rows, 1);
        i = 0;
        loop {
            if i == n {
                break;
            }
            S.set(i, 1, FP16x16 { mag: 0, sign: false });
            i += 1;
        };

        i = n;
        loop {
            if i == 0 {
                break;
            }
            let mut X_i = y.get(i - 1, 0).unwrap();
            let mut j = i;
            loop {
                if j == n {
                    break;
                }
                X_i -= X.get(i - 1, j).unwrap() * S.get(j, 0).unwrap();
                
                j += 1;
            };
            X_i /= X.get(i - 1, i - 1).unwrap();
            S.set(i - 1, 0, X_i);

            i -= 1;
        };

        return S;
    }

fn exponential_weights(lambda_unscaled: u32, l: u32) -> MutMatrix<FP16x16> {
    let lambda = FixedTrait::<FP16x16>::new_unscaled(lambda_unscaled, false) / FixedTrait::<FP16x16>::new_unscaled(100, false);
    let mut weights = MutMatrixImpl::<FP16x16>::new(l, 1);
    let mut i = 0;
    loop {
        if i == l {
            break;
        }
        let mut l_i = FixedTrait::<FP16x16>::new_unscaled(i, false);
        let mut l_pow = FixedTrait::pow(lambda, l_i);
        let mut w_i = (FixedTrait::<FP16x16>::new_unscaled(1, false) - lambda) * l_pow;
        weights.set(i, 0, w_i);
        i += 1;
    };
    return weights;
}

fn diagonalize(ref X: MutMatrix<FP16x16>) -> MutMatrix<FP16x16> {
    assert(X.rows > 1 && X.cols == 1, 'X not row vector');

    let mut i = 0;
    let mut result = MutMatrixImpl::<FP16x16>::new(X.rows, X.rows);
    loop {
        if i == X.rows {
            break;
        }
        let mut j = 0;
        loop {
            if j == X.rows {
                break;
            }
            if i == j {
                result.set(i, j, X.get(i, 0).unwrap());
            }
            else {
                result.set(i, j, FixedTrait::<FP16x16>::new(0, false));
            }
            j += 1;
        };
        i += 1;
    };
    return result;
}

fn mean(ref X: MutMatrix<FP16x16>, axis: u32) -> MutMatrix<FP16x16> {
    // Simple average case for a matrix along specified axis
    // Returns a matrix with shape (X.rows, 1) if axis == 0 or (1, X.cols) if axis == 1

    assert(axis == 0 || axis == 1, 'Wrong axis');

    // Simple average case for a vector
    // Returns a matrix with shape (1,1)
    if X.cols == 1 || X.rows == 1 {

        let mut result = MutMatrixImpl::<FP16x16>::new(1, 1);
        let mut num = FixedTrait::<FP16x16>::new_unscaled(0, false);
        let mut i = 0;

        let den = FixedTrait::<FP16x16>::new_unscaled(X.data.len(), false);

        if X.cols == 1 && X.rows == 1 {
            result.set(0, 0, X.get(0, 0).unwrap());
            return result;
        }
        else {
            let l = if (X.rows > 1) {
                X.rows
            }
            else {
                X.cols
            };
            loop {
                if i == l {
                    break;
                }
                let mut num_i = if (X.rows > 1) {
                    X.get(i, 0).unwrap()
                }
                else {
                    X.get(0, i).unwrap()
                };
                num += num_i;
                i += 1;
            };
            result.set(0, 0, num / den);
        }

        return result;
    }

    // Matrix average along specified axis
    // Returns a vector with shape=(X.rows, 1) if axis == 0 or shape=(1, X.cols) if axis == 1
    else {
        // Average along rows
        if axis == 0 {
            let mut result = MutMatrixImpl::<FP16x16>::new(X.rows, 1);
            let den = FixedTrait::<FP16x16>::new_unscaled(X.cols, false);
            let mut row = 0;

            loop {
                if row == X.rows {
                    break;
                } 
                let mut row_num = FixedTrait::<FP16x16>::new_unscaled(0, false);
                let mut col = 0;
                loop {
                    if col == X.cols {
                        break;
                    }
                    row_num += X.get(row, col).unwrap();
                    col += 1;
                };
                result.set(row, 0, row_num / den);
                row += 1;
            };

            return result;
        }

        // Average along columns
        else {
            let mut result = MutMatrixImpl::<FP16x16>::new(1, X.cols);
            let den = FixedTrait::<FP16x16>::new_unscaled(X.rows, false);
            let mut col = 0;

            loop {
                if col == X.cols {
                    break;
                } 
                let mut col_num = FixedTrait::<FP16x16>::new_unscaled(0, false);
                let mut row = 0;
                loop {
                    if row == X.rows {
                        break;
                    }
                    col_num += X.get(row, col).unwrap();
                    row += 1;
                };
                result.set(0, col, col_num / den);
                col += 1;
            };
        
            return result;
        }
    }
}

fn mean_weighted(ref X: MutMatrix<FP16x16>, ref weights: MutMatrix<FP16x16>, axis: u32) -> MutMatrix<FP16x16> {
    // Weighted average
    // Returns a matrix with shape (X.rows, 1) if axis == 0 or (1, X.cols) if axis == 1

    // Weight assertions
    if X.rows > 1 {
        assert(weights.rows == X.rows && weights.cols == 1, 'Weights shape mismatch');
    }
    else {
        assert(weights.cols == X.cols && weights.rows == 1, 'Weights shape mismatch');
    }

    // Vector case
    if X.rows == 1 || X.cols == 1 {

        assert(X.rows != X.cols, '1 element input');
    
        let mut result = MutMatrixImpl::<FP16x16>::new(1, 1);
        let mut num = FixedTrait::<FP16x16>::new_unscaled(0, false);
        let mut i = 0;

        let den = FixedTrait::<FP16x16>::new_unscaled(X.data.len(), false);
        
        let l = if (X.rows > 1) {
                X.rows
            }
            else {
                X.cols
            };
        
        loop {
            if i == l {
                break;
            }
            let mut num_i = if (X.rows > 1) {
                X.get(i, 0).unwrap() * weights.get(i, 0).unwrap()
            }
            else {
                X.get(0, i).unwrap() * weights.get(0, i).unwrap()
            };
            num += num_i;
            i += 1;
        };
        result.set(0, 0, num / den);
        
        return result;
    }

    // Matrix case
    else {
        assert(axis == 0 || axis == 1, 'Wrong axis');        

        // Average along rows
        if axis == 0 {

            let mut result = MutMatrixImpl::<FP16x16>::new(X.rows, 1);
            let mut row = 0;

            loop {
                if row == X.rows {
                    break;
                }
                let mut row_num = FixedTrait::<FP16x16>::new_unscaled(0, false);
                let mut col = 0;
                loop {
                    if col == X.cols {
                        break;
                    }
                    row_num += X.get(row, col).unwrap() * weights.get(col, 0).unwrap();
                    col += 1;
                };
                result.set(row, 0, row_num);
                row += 1;
            };

            return result;
        }

        // Average along columns
        else {
            let mut result = MutMatrixImpl::<FP16x16>::new(1, X.cols);
            let mut col = 0;

            loop {
                if col == X.cols {
                    break;
                }
                let mut col_num = FixedTrait::<FP16x16>::new_unscaled(0, false);
                let mut row = 0;
                loop {
                    if row == X.rows {
                        break;
                    }
                    col_num += X.get(row, col).unwrap() * weights.get(row, 0).unwrap();
                    row += 1;
                };
                result.set(0, col, col_num);
                col += 1;
            };

            return result;
        }
    }
}

fn covariance(ref X: MutMatrix<FP16x16>) -> MutMatrix<FP16x16> {
    assert(X.rows > 1 && X.cols > 1, 'Not enough obs');
    
    let m = X.rows; // Num observations
    let n = X.cols; // Num variables
    let mut Cov_X = MutMatrixImpl::<FP16x16>::new(n, n);
    let mut Mean_X = mean(ref X, 1);

    let mut i = 0;
    loop {
        if i == n {
            break;
        }
        let mut j = 0;
        loop {
            if j == n {
                break;
            }
            let mut k = 0;
            let mut Cov_X_i_j = FixedTrait::<FP16x16>::new_unscaled(0, false);
            loop {
                if k == m {
                    break;
                }
                Cov_X_i_j += ((X.get(k, i).unwrap() - Mean_X.get(0, i).unwrap()) * (X.get(k, j).unwrap() - Mean_X.get(0, j).unwrap()));       
                k += 1;
            };
            Cov_X_i_j /= FixedTrait::<FP16x16>::new_unscaled(m - 1, false);
            Cov_X.set(i, j, Cov_X_i_j);
            Cov_X.set(j, i, Cov_X_i_j);
            j += 1;
        };
        i += 1;
    };

    return Cov_X;
}

fn covariance_weighted(ref X: MutMatrix<FP16x16>, ref weights: MutMatrix<FP16x16>) -> MutMatrix<FP16x16> {
    assert(X.rows > 1 && X.cols > 1, 'Not enough obs');
    assert(weights.rows == X.rows && weights.cols == 1, 'Weights shape mismatch');

    // Normalize weights
    let mut normalized_weights = MutMatrixImpl::<FP16x16>::new(weights.rows, 1);
    let mut total_weight = weights.reduce_sum(0, false);
    let mut i = 0;
    loop {
        if i == weights.rows {
            break;
        }
        let w_i = weights.get(i, 0).unwrap();
        normalized_weights.set(i, 0, w_i / total_weight.get(0, 0).unwrap());
        i += 1;
    };

    let m = X.rows; // Num observations
    let n = X.cols; // Num variables
    let mut Cov_X = MutMatrixImpl::<FP16x16>::new(n, n);
    let mut Mean_X = mean_weighted(ref X, ref weights, 1);

    let mut adj_weight_sum = FixedTrait::<FP16x16>::new_unscaled(0, false);
    i = 0;
    loop {
        if i == normalized_weights.rows {
            break;
        }
        let mut w_i = normalized_weights.get(i, 0).unwrap();
        adj_weight_sum += (w_i * w_i);
        i += 1;
    };

    i = 0;
    loop {
        if i == n {
            break;
        }
        let mut j = 0;
        loop {
            if j == n {
                break;
            }
            let mut k = 0;
            let mut Cov_X_i_j = FixedTrait::<FP16x16>::new_unscaled(0, false);
            loop {
                if k == m {
                    break;
                }
                Cov_X_i_j += (normalized_weights.get(k, 0).unwrap() * (X.get(k, i).unwrap() - Mean_X.get(0, i).unwrap()) * (X.get(k, j).unwrap() - Mean_X.get(0, j).unwrap()));       
                k += 1;
            };
            let mut den = FixedTrait::<FP16x16>::new_unscaled(1, false) - adj_weight_sum;
            Cov_X_i_j /= den;
            Cov_X.set(i, j, Cov_X_i_j);
            Cov_X.set(j, i, Cov_X_i_j);
            j += 1;
        };
        i += 1;
    };

    return Cov_X;
}

fn main() {
    
    // let mut X_data = VecTrait::<NullableVec, FP16x16>::new();
    // X_data.push(FP16x16 { mag: 131072, sign: false }); // 2
    // X_data.push(FP16x16 { mag: 65536, sign: false }); // 1
    // X_data.push(FP16x16 { mag: 65536, sign: true }); // -1
    // X_data.push(FP16x16 { mag: 196608, sign: true }); // -3
    // X_data.push(FP16x16 { mag: 65536, sign: true }); // -1
    // X_data.push(FP16x16 { mag: 131072, sign: false }); // 2
    // X_data.push(FP16x16 { mag: 131072, sign: true }); // -2
    // X_data.push(FP16x16 { mag: 65536, sign: false }); // 1
    // X_data.push(FP16x16 { mag: 131072, sign: false }); // 1
    // let mut X = MutMatrix { data: X_data, rows: 3, cols: 3};

    // let mut y_data = VecTrait::<NullableVec, FP16x16>::new();
    // y_data.push(FP16x16 { mag: 524288, sign: false }); // 8
    // y_data.push(FP16x16 { mag: 720896, sign: true }); // -11
    // y_data.push(FP16x16 { mag: 196608, sign: true }); // -3
    // let mut y = MutMatrix { data: y_data, rows: 3, cols: 1};

    // Linalg test
    // let mut S = linalg_solve(ref X, ref y);
    // test_matrix(ref S);
    // Solution is [2, 3, -1] in FP16x16 format!

    // let mut Y1_data = VecTrait::<NullableVec, FP16x16>::new();
    // Y1_data.push(FP16x16 { mag: 131072, sign: false }); // 2
    // Y1_data.push(FP16x16 { mag: 65536, sign: false }); // 1
    // Y1_data.push(FP16x16 { mag: 393216, sign: false }); // 6
    // let mut Y1 = MutMatrix { data: Y1_data, rows: 3, cols: 1};

    // let mut Y2_data = VecTrait::<NullableVec, FP16x16>::new();
    // Y2_data.push(FP16x16 { mag: 65536, sign: false }); // 1
    // Y2_data.push(FP16x16 { mag: 131072, sign: false }); // 2
    // Y2_data.push(FP16x16 { mag: 196608, sign: false }); // 3
    // Y2_data.push(FP16x16 { mag: 262144, sign: false }); // 4
    // Y2_data.push(FP16x16 { mag: 327680, sign: false }); // 5
    // Y2_data.push(FP16x16 { mag: 393216, sign: false }); // 6
    // Y2_data.push(FP16x16 { mag: 458752, sign: false }); // 7
    // Y2_data.push(FP16x16 { mag: 524288, sign: false }); // 8
    // Y2_data.push(FP16x16 { mag: 589824, sign: false }); // 9
    // let mut Y2 = MutMatrix { data: Y2_data, rows: 3, cols: 3};

    // let mut weights_data = VecTrait::<NullableVec, FP16x16>::new();
    // weights_data.push(FP16x16 { mag: 13017, sign: false }); // 0.2
    // weights_data.push(FP16x16 { mag: 19661, sign: false }); // 0.3
    // weights_data.push(FP16x16 { mag: 32768, sign: false }); // 0.5
    // let mut W = MutMatrix { data: weights_data, rows: 3, cols: 1};

    // Mean tests
    // let mut Y1_mean = mean(ref Y1);
    // let mut Y2_mean_rows = mean_matrix(ref Y2, 0);
    // let mut Y2_mean_cols = mean_matrix(ref Y2, 1);
    // 'Y1_mean:'.print();
    // test_matrix(ref Y1_mean);
    // 'Y2_mean_rows:'.print();
    // test_matrix(ref Y2_mean_rows);
    // 'Y2_mean_cols:'.print();
    // test_matrix(ref Y2_mean_cols);
    
    // let mut Y1_mean_weighted = mean_weighted(ref Y1, ref W, 0);
    // let mut Y2_mean_weighted_rows = mean_weighted(ref Y2, ref W, 0);
    // let mut Y2_mean_weighted_cols = mean_weighted(ref Y2, ref W, 1);
    // 'Y1_mean_weighted:'.print();
    // test_matrix(ref Y1_mean_weighted);
    // 'Y2_mean_weighted_rows:'.print();
    // test_matrix(ref Y2_mean_weighted_rows);
    // 'Y2_mean_weighted_cols:'.print();
    // test_matrix(ref Y2_mean_weighted_cols);

    // let mut W_diag = diagonalize(ref W);
    // test_matrix(ref W_diag);

    // let mut Y3_data = VecTrait::<NullableVec, FP16x16>::new();
    // Y3_data.push(FP16x16 { mag: 65536, sign: false }); // 1
    // Y3_data.push(FP16x16 { mag: 131072, sign: false }); // 2
    // Y3_data.push(FP16x16 { mag: 196608, sign: false }); // 3
    // Y3_data.push(FP16x16 { mag: 262144, sign: false }); // 4
    // let mut Y3 = MutMatrix { data: Y3_data, rows: 2, cols: 2};

    // let mut Y4_data = VecTrait::<NullableVec, FP16x16>::new();
    // Y4_data.push(FP16x16 { mag: 65536, sign: false }); // 1
    // Y4_data.push(FP16x16 { mag: 131072, sign: false }); // 2
    // Y4_data.push(FP16x16 { mag: 196608, sign: false }); // 3
    // Y4_data.push(FP16x16 { mag: 262144, sign: false }); // 4
    // Y4_data.push(FP16x16 { mag: 327680, sign: false }); // 5
    // Y4_data.push(FP16x16 { mag: 393216, sign: false }); // 6
    // let mut Y4 = MutMatrix { data: Y4_data, rows: 2, cols: 3};

    // let mut Y = MutMatrixImpl::<FP16x16>::new(3,1);
    // Y.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    // Y.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    // Y.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    // Y.set(1, 1, FixedTrait::<FP16x16>::new_unscaled(4, false));
    // Y.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(5, false));
    // Y.set(2, 1, FixedTrait::<FP16x16>::new_unscaled(6, false));

    let mut X = MutMatrixImpl::<FP16x16>::new(4,3);
    X.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    X.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    X.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(3, false));
    X.set(3, 0, FixedTrait::<FP16x16>::new_unscaled(4, false));
    X.set(0, 1, FixedTrait::<FP16x16>::new_unscaled(5, false));
    X.set(1, 1, FixedTrait::<FP16x16>::new_unscaled(6, false));
    X.set(2, 1, FixedTrait::<FP16x16>::new_unscaled(7, false));
    X.set(3, 1, FixedTrait::<FP16x16>::new_unscaled(8, false));
    X.set(0, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    X.set(1, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    X.set(2, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));
    X.set(3, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));

    let mut weights = MutMatrixImpl::<FP16x16>::new(4, 1);
    weights.set(0, 0, FixedTrait::<FP16x16>::new(6554, false)); // 0.1
    weights.set(1, 0, FixedTrait::<FP16x16>::new(13107, false)); // 0.2
    weights.set(2, 0, FixedTrait::<FP16x16>::new(19661, false)); // 0.3
    weights.set(3, 0, FixedTrait::<FP16x16>::new(26214, false)); // 0.4

    let mut sigma2 = covariance_weighted(ref X, ref weights);
    test_matrix(ref sigma2);
    
    }