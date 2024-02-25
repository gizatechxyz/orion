use core::array::ArrayTrait;
use core::option::OptionTrait;
use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::implementations::fp16x16::math::core::{ceil, abs};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FixedTrait, FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl};
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
use orion::operators::matrix::matrix::{MutMatrix, MutMatrixTrait, MutMatrixImpl};

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

