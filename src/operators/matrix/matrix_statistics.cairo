use core::array::ArrayTrait;
use core::option::OptionTrait;
use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::implementations::fp16x16::math::core::{ceil, abs};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FixedTrait, FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl};
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
use orion::operators::matrix::matrix::{MutMatrix, MutMatrixTrait, MutMatrixImpl};

/// Trait
trait MatrixStatisticsTrait<T> {
    /// # MatrixStatisticsTrait::exponential_weights
    ///
    /// ```rust 
    ///    fn exponential_weights(lambda_unscaled: u32, l: u32) -> MutMatrix<FP16x16>
    /// ```
    /// 
    /// ## Args
    ///
    /// * `lambda_unscaled`: Input u32 number from 0 to 100
    /// * `l`: length of output vector
    ///
    /// ## Returns
    ///
    /// * Output vector of length l
    ///
    /// ## Examples
    /// let mut X = exponential_weights(97, 3);
    /// Output:
    /// [FP16x16 { mag: 1967, sign: false }, FP16x16 { mag: 1907, sign: false }, FP16x16 { mag: 1850, sign: false }]
    
    fn exponential_weights(lambda_unscaled: u32, l: u32) -> MutMatrix<T> ;
    
    /// # MatrixStatisticsTrait::mean
    ///
    /// ```rust 
    ///    fn mean(ref X: MutMatrix<FP16x16>, axis: u32) -> MutMatrix<FP16x16>
    /// ```
    /// 
    /// ## Args
    ///
    /// * `X`: Input 1D vector or 2D matrix
    /// * `axis`: 0 for rows, 1 for columns
    ///
    /// ## Returns
    ///
    /// * Simple mean for a vector input, vector of means for a matrix input along specified axis.
    ///
    /// ## Type Constraints
    ///
    /// X must be FP16x16 valued.
    ///
    /// ## Examples
    ///
    /// let mut Y = MutMatrixTrait::<FP16x16>::new(3,1);
    /// Y.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    /// Y.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// Y.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(3, false));
    /// let mut mu_Y = mean(ref Y, 0);
    ///
    /// let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
    /// X.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    /// X.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(3, false));
    /// X.set(3, 0, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// X.set(0, 1, FixedTrait::<FP16x16>::new_unscaled(5, false));
    /// X.set(1, 1, FixedTrait::<FP16x16>::new_unscaled(6, false));
    /// X.set(2, 1, FixedTrait::<FP16x16>::new_unscaled(7, false));
    /// X.set(3, 1, FixedTrait::<FP16x16>::new_unscaled(8, false));
    /// X.set(0, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(1, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// X.set(3, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// let mut mu1_X = mean(ref X, 0);
    /// let mut mu2_X = mean(ref X, 1);
    ///
    /// Output:
    /// mu_Y: [FP16x16 { mag: 13107, sign: false }]
    /// mu1_X: [FP16x16 { mag: 174762, sign: false }, FP16x16 { mag: 218453, sign: false }, FP16x16 { mag: 305834, sign: false }, FP16x16 { mag: 349525, sign: false }]
    /// mu2_X: [FP16x16 { mag: 163840, sign: false }, FP16x16 { mag: 425984, sign: false }, FP16x16 { mag: 196608, sign: false }]
    
    fn mean(ref X: MutMatrix<T>, axis: u32) -> MutMatrix<T>;

    /// # MatrixStatisticsTrait::mean_weighted
    ///
    /// ```rust 
    ///    fn mean_weighted(ref X: MutMatrix<FP16x16>, ref weights: MutMatrix<FP16x16>, axis: u32) -> MutMatrix<FP16x16>
    /// ```
    /// 
    /// ## Args
    ///
    /// * `X`: Input 1D vector or 2D matrix
    /// * `weights`: Input 1D row vector of weights
    /// * `axis`: 0 for rows, 1 for columns
    ///
    /// ## Returns
    ///
    /// * Weighted mean for a vector input, vector of weighted means for a matrix input along specified axis.
    ///
    /// ## Type Constraints
    ///
    /// X must be FP16x16 valued
    /// weights must be FP16x16 valued and a row vector
    ///
    /// ## Examples
    /// let mut weights = MutMatrixTrait::<FP16x16>::new(4,1);
    /// weights.set(0, 0, FixedTrait::<FP16x16>::new(6554, false));
    /// weights.set(1, 0, FixedTrait::<FP16x16>::new(13107, false));
    /// weights.set(2, 0, FixedTrait::<FP16x16>::new(19661, false));
    ///
    /// let mut weights2 = MutMatrixTrait::<FP16x16>::new(4,1);
    /// weights2.set(0, 0, FixedTrait::<FP16x16>::new(6554, false));
    /// weights2.set(1, 0, FixedTrait::<FP16x16>::new(13107, false));
    /// weights2.set(2, 0, FixedTrait::<FP16x16>::new(19661, false));
    /// weights2.set(3, 0, FixedTrait::<FP16x16>::new(26214, false));
    ///
    /// let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
    /// X.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    /// X.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(3, false));
    /// X.set(3, 0, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// X.set(0, 1, FixedTrait::<FP16x16>::new_unscaled(5, false));
    /// X.set(1, 1, FixedTrait::<FP16x16>::new_unscaled(6, false));
    /// X.set(2, 1, FixedTrait::<FP16x16>::new_unscaled(7, false));
    /// X.set(3, 1, FixedTrait::<FP16x16>::new_unscaled(8, false));
    /// X.set(0, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(1, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// X.set(3, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// let mut mu1_X = mean(ref X, ref weights, 0);
    /// let mut mu2_X = mean(ref X, ref weights2, 1);
    ///
    /// Output:
    /// mu1_X: [FP16x16 { mag: 163839, sign: false }, FP16x16 { mag: 183500, sign: false }, FP16x16 { mag: 294911, sign: false }, FP16x16 { mag: 314572, sign: false }]
    /// mu2_X: [FP16x16 { mag: 196607, sign: false }, FP16x16 { mag: 458751, sign: false }, FP16x16 { mag: 222822, sign: false }]
    
    fn mean_weighted(ref X: MutMatrix<T>, ref weights: MutMatrix<T>, axis: u32) -> MutMatrix<T>;

    /// # MatrixStatisticsTrait::covariance
    ///
    /// ```rust 
    ///    fn covariance(ref X: MutMatrix<FP16x16>) -> MutMatrix<FP16x16>;
    /// ```
    /// 
    /// ## Args
    ///
    /// * `X`: Input 1D vector or 2D matrix
    ///
    /// ## Returns
    ///
    /// * (n,n) covariance matrix for (m,n) matrix input.
    ///
    /// ## Type Constraints
    ///
    /// X must be FP16x16 valued.
    /// Assumes columns are variables and rows are observations
    ///
    /// ## Examples
    ///
    /// let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
    /// X.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    /// X.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(3, false));
    /// X.set(3, 0, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// X.set(0, 1, FixedTrait::<FP16x16>::new_unscaled(5, false));
    /// X.set(1, 1, FixedTrait::<FP16x16>::new_unscaled(6, false));
    /// X.set(2, 1, FixedTrait::<FP16x16>::new_unscaled(7, false));
    /// X.set(3, 1, FixedTrait::<FP16x16>::new_unscaled(8, false));
    /// X.set(0, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(1, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// X.set(3, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));
    ///
    /// let mut sigma2_X = covariance(ref X);
    ///
    /// Output:
    /// [[FP16x16 { mag: 109226, sign: false }, FP16x16 { mag: 109226, sign: false }, FixedTrait::<FP16x16>::new(87381, false)],
    ///  [FP16x16 { mag: 109226, sign: false }, FP16x16 { mag: 109226, sign: false }, FixedTrait::<FP16x16>::new(87381, false)],
    ///  [FP16x16 { mag: 87381, sign: false }, FP16x16 { mag: 87381, sign: false }, FixedTrait::<FP16x16>::new(87381, false)]]

    fn covariance(ref X: MutMatrix<T>) -> MutMatrix<T>;

    /// # MatrixStatisticsTrait::covariance_weighted
    ///
    /// ```rust 
    ///    fn covariance_weighted(ref X: MutMatrix<FP16x16>, ref weights: MutMatrix<FP16x16>) -> MutMatrix<FP16x16>;
    /// ```
    /// 
    /// ## Args
    ///
    /// * `X`: Input 1D vector or 2D matrix
    /// * `weights`: Input 1D row vector of weights
    ///
    /// ## Returns
    ///
    /// * (n,n) weighted covariance matrix for (m,n) matrix input and (m, 1) weights input.
    ///
    /// ## Type Constraints
    ///
    /// X and weights must be FP16x16 valued.
    /// Assumes columns are variables and rows are observations.
    ///
    /// ## Examples
    ///
    /// let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
    /// X.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    /// X.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(3, false));
    /// X.set(3, 0, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// X.set(0, 1, FixedTrait::<FP16x16>::new_unscaled(5, false));
    /// X.set(1, 1, FixedTrait::<FP16x16>::new_unscaled(6, false));
    /// X.set(2, 1, FixedTrait::<FP16x16>::new_unscaled(7, false));
    /// X.set(3, 1, FixedTrait::<FP16x16>::new_unscaled(8, false));
    /// X.set(0, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(1, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));
    /// X.set(3, 2, FixedTrait::<FP16x16>::new_unscaled(4, false));
    ///
    /// let mut weights = MutMatrixTrait::<FP16x16>::new(4,1);
    /// weights.set(0, 0, FixedTrait::<FP16x16>::new(6554, false)); // 0.1
    /// weights.set(1, 0, FixedTrait::<FP16x16>::new(13107, false)); // 0.2
    /// weights.set(2, 0, FixedTrait::<FP16x16>::new(19661, false)); // 0.3
    /// weights.set(3, 0, FixedTrait::<FP16x16>::new(26214, false)); // 0.4
    ///
    /// let mut sigma2_X = covariance(ref X, ref weights);
    ///
    /// Output:
    /// [[FP16x16 { mag: 93613, sign: false }, FP16x16 { mag: 93613, sign: false }, FixedTrait::<FP16x16>::new(74889, false)],
    ///  [FP16x16 { mag: 93613, sign: false }, FP16x16 { mag: 93613, sign: false }, FixedTrait::<FP16x16>::new(74889, false)],
    ///  [FP16x16 { mag: 74889, sign: false }, FP16x16 { mag: 74889, sign: false }, FixedTrait::<FP16x16>::new(78632, false)],]

    fn covariance_weighted(ref X: MutMatrix<T>, ref weights: MutMatrix<T>) -> MutMatrix<T>;
}

impl MatrixStatisticsImpl of MatrixStatisticsTrait<FP16x16> {
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
        let mut Mean_X = MatrixStatisticsImpl::mean(ref X, 1);

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
        let mut Mean_X = MatrixStatisticsImpl::mean_weighted(ref X, ref weights, 1);

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
}

