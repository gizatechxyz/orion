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
trait MatrixLinalgTrait<FP16x16> {
    /// # MatrixLinalgTrait::diagonalize
    ///
    /// ```rust 
    ///    fn diagonalize(ref X: MutMatrix<T>) -> MutMatrix<T>;
    /// ```
    /// 
    /// ## Args
    ///
    /// * `X`: Input 1D vector
    ///
    /// ## Returns
    ///
    /// * 2D diagonal matrix.
    ///
    /// ## Type Constraints
    ///
    /// X must be FP16x16 valued.
    ///
    /// ## Examples
    /// let mut X = MutMatrixImpl::<FP16x16>::new(3, 1);
    /// X.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    /// X.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(3, false));
    /// let mut diag_X = diagonalize(ref X);
    /// Output:
    /// [[FP16x16 { mag: 65536, sign: false }, 0, 0], [0, FP16x16 { mag: 13107, sign: false }, 0], [0, 0, FP16x16 { mag: 196608, sign: false }]]
    
    fn diagonalize(ref X: MutMatrix<FP16x16>) -> MutMatrix<FP16x16>;
    
    /// # MatrixLinalgTrait::linalg_solve
    ///
    /// ```rust 
    ///    fn(ref X: MutMatrix<T>, ref y: MutMatrix<T>) -> MutMatrix<T>;
    /// ```
    /// 
    /// ## Args
    ///
    /// * `X`: Input 2D MutMatrix
    /// * `y`: Input 1D vector
    ///
    /// ## Returns
    ///
    /// * 1D solution vector
    ///
    /// ## Type Constraints
    ///
    /// X, y must be FP16x16 valued.
    ///
    /// ## Examples
    /// let mut X = MutMatrixTrait::<FP16x16>::new(3, 3);
    /// X.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(0, 1, FixedTrait::<FP16x16>::new_unscaled(1, false));
    /// X.set(0, 2, FixedTrait::<FP16x16>::new_unscaled(1, true));
    /// X.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(3, true));
    /// X.set(1, 1, FixedTrait::<FP16x16>::new_unscaled(1, true));
    /// X.set(1, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    /// X.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(2, true));
    /// X.set(2, 1, FixedTrait::<FP16x16>::new_unscaled(1, false));
    /// X.set(2, 2, FixedTrait::<FP16x16>::new_unscaled(1, false));
    /// let mut Y = MutMatrixTrait::<FP16x16>::new(3, 1);
    /// Y.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(8, false));
    /// Y.set(0, 1, FixedTrait::<FP16x16>::new_unscaled(11, true));
    /// Y.set(0, 2, FixedTrait::<FP16x16>::new_unscaled(3, true));
    /// let mut S = linalg_solve(ref X, ref Y);
    /// Output:
    /// [FP16x16 { mag: 131072, sign: false }, FP16x16 { mag: 196608, sign: false }, FP16x16 { mag: 65536, sign: true }]

    fn linalg_solve(ref X: MutMatrix<FP16x16>, ref y: MutMatrix<FP16x16>) -> MutMatrix<FP16x16>;
}

impl MatrixLinalgImpl<
    FP16x16,
    MAG, 
    +Drop<FP16x16>, 
    +Copy<FP16x16>, 
    +NumberTrait<FP16x16, MAG>, 
    +PartialOrd<FP16x16>, 
    +FixedTrait<FP16x16, MAG>,
    +MutMatrix<FP16x16>,
    +Add<FP16x16>,
    +AddEq<FP16x16>,
    +Sub<FP16x16>, 
    +SubEq<FP16x16>,
    +Mul<FP16x16>,
    +MulEq<FP16x16>,
    +Div<FP16x16>,
    +DivEq<FP16x16>> of MatrixLinalgTrait<FP16x16> {
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
                    result.set(i, j, FixedTrait::new(0, false));
                }
                j += 1;
            };
            i += 1;
        };
        return result;
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
}