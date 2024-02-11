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

// fn weights(lambda_unscaled: u32, l: u32) -> MutMatrix<FP16x16> {
//         let mut lambda = FixedTrait::<FP16x16>::new_unscaled(lambda_unscaled, false) / FixedTrait::<FP16x16>::new_unscaled(100, false); 
//         let mut weights = MutMatrixImpl::new(l, 1);
//         let mut i: u32 = 0;
//         loop {
//             if i == l {
//                 break ();
//             }
//             let mut i_fp = FixedTrait::<FP16x16>::new_unscaled(i, false);
//             let mut x = (FixedTrait::<FP16x16>::new_unscaled(1, false) - lambda) * (lambda.pow(i_fp));
//             weights.set(i, 0, x);
//             i += 1;
//         };

//         return weights;
//     }

fn mean(ref X: MutMatrix<FP16x16>, weights: Option<MutMatrix<FP16x16>>, axis: Option<u32>) -> MutMatrix<FP16x16> {
    let mut i = 0;
    
    // Assume by default that the weights are a column vector
    match weights {
        Option::None => {
            let mut weights = MutMatrixImpl:new()
        },
        Option::Some(x) => {
            assert(weights.rows == X.rows & weights.cols == 1, 'Wrong shape weights');
        },
    }

    // Vector average
    match axis {
        Option::None => {
            assert(X.rows == 1 || X.cols == 1, 'Wrong shape');

            if X.cols == 1 & X.rows != 1 {
                let mut X = X.transpose();
            }

            let mut result = MutMatrixImpl::new(1, 1);
            let mut num = FixedTrait::new_unscaled(0, false);
            let l = FixedTrait::new_unscaled(mag: X.data.len(), sign: false);
            loop {
                if i == X.rows {
                    break;
                }
                if weights == Option::None {
                    num += X.get(i, 0);
                }
                else {
                    num += X.get(i, 0) * Option::Some(weights).get(i, 0);
                }
                i += 1;
            };
            result.set(0, 0, num / l);
        },

        // Matrix average along specified axis
        Option::Some(x) => {
            assert(axis == 0 || axis == 1, 'Wrong axis');
            
            // Average along rows
            if axis == 0 {
                let mut result = MutMatrixImpl::new(X.rows, 1);
                let l = FixedTrait::new_unscaled(mag: X.data.len() / X.rows, sign: false);
                loop {
                    if i == X.rows {
                        break;
                    }
                    let mut num = FixedTrait::new_unscaled(0, false);
                    let mut j = 0;
                    loop {
                        if j == X.cols {
                            break;
                        }
                        
                        if weights == Option::None {
                            num += X.get(i, j);
                        }
                        else {
                            // Assert weights are a row vector
                            num += X.get(i, j) * Option::Some(weights).get(j, 0); // Need to assert shape correctly 
                        }

                        j += 1;
                    };
                    result.set(i, 0, num / l);

                    i += 1;
                };
            }

            // Average along columns
            else {
                let mut result = MutMatrixImpl::new(1, X.cols);
                let l = FixedTrait::new_unscaled(mag: X.data.len() / X.cols, sign: false);
                loop {
                    if i == X.cols {
                        break;
                    }
                    let mut num = FixedTrait::new_unscaled(0, false);
                    let mut j = 0;
                    loop {
                        if j == X.rows {
                            break;
                        }
                        
                        if weights == Option::None {
                            num += X.get(j, i);
                        }
                        else {
                            num += X.get(j, i) * Option::Some(weights).get(j, 0); 
                        }

                        j += 1;
                    };
                    result.set(0, i, num / l);

                    i += 1;
                };
            }
        },
    }

    return result;
}

// fn cov(X: MutMatrix<FP16x16>, weights: MutMatrix<FP16x16>) -> MutMatrix<FP16x16> {
//     0
// }

// fn diag(X: MutMatrix<FP16x16>) -> MutMatrix<FP16x16> {
//     0
// }

fn main(){
    
    let mut X_data = VecTrait::<NullableVec, FP16x16>::new();
    X_data.push(FP16x16 { mag: 131072, sign: false }); // 2
    X_data.push(FP16x16 { mag: 65536, sign: false }); // 1
    X_data.push(FP16x16 { mag: 65536, sign: true }); // -1
    X_data.push(FP16x16 { mag: 196608, sign: true }); // -3
    X_data.push(FP16x16 { mag: 65536, sign: true }); // -1
    X_data.push(FP16x16 { mag: 131072, sign: false }); // 2
    X_data.push(FP16x16 { mag: 131072, sign: true }); // -2
    X_data.push(FP16x16 { mag: 65536, sign: false }); // 1
    X_data.push(FP16x16 { mag: 131072, sign: false }); // 1
    let mut X = MutMatrix { data: X_data, rows: 3, cols: 3};

    let mut y_data = VecTrait::<NullableVec, FP16x16>::new();
    y_data.push(FP16x16 { mag: 524288, sign: false }); // 8
    y_data.push(FP16x16 { mag: 720896, sign: true }); // -11
    y_data.push(FP16x16 { mag: 196608, sign: true }); // -3
    let mut y = MutMatrix { data: y_data, rows: 3, cols: 1};

    // Linalg test
    // let mut S = linalg_solve(ref X, ref y);
    // test_matrix(ref S);
    // Solution is [2, 3, -1] in FP16x16 format!

    // Exp weights test
    
    let mut weights_data = VecTrait::<NullableVec, FP16x16>::new();
    weights_data.push(FP16x16 { mag: 22453, sign: false }); // ~0.34
    weights_data.push(FP16x16 { mag: 16731, sign: false }); // ~0.26
    weights_data.push(FP16x16 { mag: 26352, sign: false }); // ~0.40
    let mut W = MutMatrix { data: weights_data, rows: 3, cols: 1};
    let mut W_bad = MutMatrix { data: weights_data, rows: 1, cols: 3};

    let mut Y_data = VecTrait::<NullableVec, FP16x16>::new();
    Y_data.push(FP16x16 { mag: 131072, sign: false }); // 2
    Y_data.push(FP16x16 { mag: 65536, sign: false }); // 1
    Y_data.push(FP16x16 { mag: 393216, sign: false }); // 6
    let mut Y = MutMatrix { data: Y_data, rows: 3, cols: 1};

    // Mean test (1D)
    let Y_simple_avg = mean(ref Y, (), ()); // Should return 3_fp16x16
    // let Y_weighted_avg = mean(ref Y, weights=Option(W));
    // let Y_weighted_avg_bad = mean(ref Y, weights=Option(W_bad)); // should error out
    // 'Y smpl avg:'.print();
    // test_matrix(ref Y_simple_avg);
    // 'Y wtd avg:'.print();
    // test_matrix(ref Y_weighted_avg);

    // // Mean test (2D)
    // let Y_simple_avg_axis_0 = mean(ref X, axis=Option(0));
    // let Y_weighted_avg_axis_0 = mean(ref X, weights=Option(W), axis=Option(0));
    // let Y_simple_avg_axis_1 = mean(ref X, axis=Option(1));
    // let Y_weighted_avg_axis_1 = mean(ref X, weights=Option(W), axis=Option(1));
    // 'Y smpl avg ax=0:'.print();
    // test_matrix(ref Y_simple_avg_axis_0);
    //  'Y wtd avg ax=0:'.print();
    // test_matrix(ref Y_weighted_avg_axis_0);
    // 'Y smpl avg ax=1:'.print();
    // test_matrix(ref Y_simple_avg_axis_1);
    //  'Y wtd avg ax=1:'.print();
    // test_matrix(ref Y_weighted_avg_axis_1);

    }