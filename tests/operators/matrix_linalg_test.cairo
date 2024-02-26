use core::option::OptionTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FixedTrait, FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl};
use orion::operators::matrix::matrix::{MutMatrix, MutMatrixTrait, MutMatrixImpl};
use orion::operators::matrix::matrix_linalg::MatrixLinalgTrait;

#[test]
#[available_gas(99999999999999999)]
fn matrix_linalg_test() {
    let ERROR_THRESHOLD = FixedTrait::<FP16x16>::new_unscaled(100, false); // ~0.00153 error threshold

    let mut X = MutMatrixTrait::<FP16x16>::new(3, 3);
    X.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    X.set(0, 1, FixedTrait::<FP16x16>::new_unscaled(1, false));
    X.set(0, 2, FixedTrait::<FP16x16>::new_unscaled(1, true));
    X.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(3, true));
    X.set(1, 1, FixedTrait::<FP16x16>::new_unscaled(1, true));
    X.set(1, 2, FixedTrait::<FP16x16>::new_unscaled(2, false));
    X.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(2, true));
    X.set(2, 1, FixedTrait::<FP16x16>::new_unscaled(1, false));
    X.set(2, 2, FixedTrait::<FP16x16>::new_unscaled(1, false));

    let mut Y = MutMatrixTrait::<FP16x16>::new(3, 1);
    Y.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(8, false));
    Y.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(11, true));
    Y.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(3, true));
    
    let mut S = MatrixLinalgTrait::<FP16x16>::linalg_solve(ref X, ref Y);
    
    // Solution = [2, 3, -1]

    assert(S.rows == X.rows, 'Wrong num of rows');
    assert(S.cols == 1, 'Wrong num of cols');

    // Check mags
    assert(FixedTrait::abs(S.get(0, 0).unwrap() - FixedTrait::<FP16x16>::new_unscaled(2, false)) < ERROR_THRESHOLD, 'S_1 mag incorrect'); // ~2
    assert(FixedTrait::abs(S.get(1, 0).unwrap() - FixedTrait::<FP16x16>::new_unscaled(3, false)) < ERROR_THRESHOLD, 'S_2 mag incorrect'); // ~3
    assert(FixedTrait::abs(S.get(2, 0).unwrap() - FixedTrait::<FP16x16>::new_unscaled(1, true)) < ERROR_THRESHOLD, 'S_3 mag incorrect'); // ~-1
    
    // Check signs
    assert(S.get(0, 0).unwrap().sign == false, 'S_1 sign incorrect');
    assert(S.get(1, 0).unwrap().sign == false, 'S_2 sign incorrect');
    assert(S.get(2, 0).unwrap().sign == true, 'S_3 sign incorrect');

}