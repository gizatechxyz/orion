use core::option::OptionTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FixedTrait, FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl};
use orion::operators::matrix::matrix::{MutMatrix, MutMatrixTrait, MutMatrixImpl};
use orion::operators::matrix::matrix_statistics::MatrixStatisticsTrait;

#[test]
#[available_gas(200000000000)]
fn exponential_weights_test() {
    let ERROR_THRESHOLD = FixedTrait::<FP16x16>::new_unscaled(100, false); // ~0.00153 error threshold

    let mut X = MatrixStatisticsTrait::<FP16x16>::exponential_weights(97, 3);

    assert(X.rows == 3 && X.cols == 1, 'Shape incorrect');
    assert(FixedTrait::abs(X.get(0, 0).unwrap() - FixedTrait::<FP16x16>::new(1967, false)) < ERROR_THRESHOLD, 'X_1 incorrect'); // ~0.300
    assert(FixedTrait::abs(X.get(1, 0).unwrap() - FixedTrait::<FP16x16>::new(1907, false)) < ERROR_THRESHOLD, 'X_2 incorrect'); // ~0.029
    assert(FixedTrait::abs(X.get(2, 0).unwrap() - FixedTrait::<FP16x16>::new(1850, false)) < ERROR_THRESHOLD, 'X_3 incorrect'); // ~0.028
}

#[test]
#[available_gas(200000000000)]
fn mean_test_1D() {
    let ERROR_THRESHOLD = FixedTrait::<FP16x16>::new_unscaled(100, false); // ~0.00153 error threshold

    let mut X = MutMatrixTrait::<FP16x16>::new(3,1);
    X.set(0, 0, FixedTrait::<FP16x16>::new_unscaled(1, false));
    X.set(1, 0, FixedTrait::<FP16x16>::new_unscaled(2, false));
    X.set(2, 0, FixedTrait::<FP16x16>::new_unscaled(3, false));

    let mut mu_X = MatrixStatisticsTrait::<FP16x16>::mean(ref X, 0);

    assert(mu_X.rows == 1 && mu_X.cols == 1, 'Shape incorrect');
    assert(FixedTrait::abs(mu_X.get(0, 0).unwrap() - FixedTrait::<FP16x16>::new_unscaled(2, false)) < ERROR_THRESHOLD, 'mean_X incorrect');
}

#[test]
#[available_gas(200000000000)]
fn mean_test_2D_i() {
    let ERROR_THRESHOLD = FixedTrait::<FP16x16>::new_unscaled(100, false); // ~0.00153 error threshold
    
    let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
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

    let mut mu_X = MatrixStatisticsTrait::<FP16x16>::mean(ref X, 0);

    assert(mu_X.rows == 4 && mu_X.cols == 1, 'Shape incorrect');
    assert(FixedTrait::abs(mu_X.get(0, 0).unwrap() - FixedTrait::<FP16x16>::new(174762, false)) < ERROR_THRESHOLD, 'mean_X_1 incorrect'); // ~2.67
    assert(FixedTrait::abs(mu_X.get(1, 0).unwrap() - FixedTrait::<FP16x16>::new(218453, false)) < ERROR_THRESHOLD, 'mean_X_2 incorrect'); // ~3.33
    assert(FixedTrait::abs(mu_X.get(2, 0).unwrap() - FixedTrait::<FP16x16>::new(305834, false)) < ERROR_THRESHOLD, 'mean_X_3 incorrect'); // ~4.67
    assert(FixedTrait::abs(mu_X.get(3, 0).unwrap() - FixedTrait::<FP16x16>::new(349525, false)) < ERROR_THRESHOLD, 'mean_X_3 incorrect'); // ~5.33
}

#[test]
#[available_gas(200000000000)]
fn mean_test_2D_ii() {
    let ERROR_THRESHOLD = FixedTrait::<FP16x16>::new_unscaled(100, false); // ~0.00153 error threshold
    
    let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
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

    let mut mu_X = MatrixStatisticsTrait::<FP16x16>::mean(ref X, 1);

    assert(mu_X.rows == 1 && mu_X.cols == 3, 'Shape incorrect');
    assert(FixedTrait::abs(mu_X.get(0, 0).unwrap() - FixedTrait::<FP16x16>::new(163840, false)) < ERROR_THRESHOLD, 'mean_X_1 incorrect'); // ~2.5
    assert(FixedTrait::abs(mu_X.get(0, 1).unwrap() - FixedTrait::<FP16x16>::new(425984, false)) < ERROR_THRESHOLD, 'mean_X_2 incorrect'); // ~6.5
    assert(FixedTrait::abs(mu_X.get(0, 2).unwrap() - FixedTrait::<FP16x16>::new(196608, false)) < ERROR_THRESHOLD, 'mean_X_3 incorrect'); // ~3
}

#[test]
#[available_gas(200000000000)]
fn mean_weighted_test_i() {
    let ERROR_THRESHOLD = FixedTrait::<FP16x16>::new_unscaled(100, false); // ~0.00153 error threshold
    
    let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
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

    let mut weights = MutMatrixTrait::<FP16x16>::new(4,1);
    weights.set(0, 0, FixedTrait::<FP16x16>::new(6554, false)); // 0.1
    weights.set(1, 0, FixedTrait::<FP16x16>::new(13107, false)); // 0.2
    weights.set(2, 0, FixedTrait::<FP16x16>::new(19661, false)); // 0.7

    let mut mu_X = MatrixStatisticsTrait::<FP16x16>::mean_weighted(ref X, ref weights, 0);

    assert(mu_X.rows == 4 && mu_X.cols == 1, 'Shape incorrect');
    assert(FixedTrait::abs(mu_X.get(0, 0).unwrap() - FixedTrait::<FP16x16>::new(163839, false)) < ERROR_THRESHOLD, 'mean_X_1 incorrect'); // ~2.5
    assert(FixedTrait::abs(mu_X.get(1, 0).unwrap() - FixedTrait::<FP16x16>::new(183500, false)) < ERROR_THRESHOLD, 'mean_X_2 incorrect'); // ~2.8
    assert(FixedTrait::abs(mu_X.get(2, 0).unwrap() - FixedTrait::<FP16x16>::new(294911, false)) < ERROR_THRESHOLD, 'mean_X_3 incorrect'); // ~4.5
    assert(FixedTrait::abs(mu_X.get(3, 0).unwrap() - FixedTrait::<FP16x16>::new(314572, false)) < ERROR_THRESHOLD, 'mean_X_4 incorrect'); // ~4.8
}

#[test]
#[available_gas(200000000000)]
fn mean_weighted_test_ii() {
    let ERROR_THRESHOLD = FixedTrait::<FP16x16>::new_unscaled(100, false); // ~0.00153 error threshold
    
    let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
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

    let mut weights = MutMatrixTrait::<FP16x16>::new(4,1);
    weights.set(0, 0, FixedTrait::<FP16x16>::new(6554, false)); // 0.1
    weights.set(1, 0, FixedTrait::<FP16x16>::new(13107, false)); // 0.2
    weights.set(2, 0, FixedTrait::<FP16x16>::new(19661, false)); // 0.3
    weights.set(3, 0, FixedTrait::<FP16x16>::new(26214, false)); // 0.4

    let mut mu_X = MatrixStatisticsTrait::<FP16x16>::mean_weighted(ref X, ref weights, 1);

    assert(mu_X.rows == 1 && mu_X.cols == 3, 'Shape incorrect');
    assert(FixedTrait::abs(mu_X.get(0, 0).unwrap() - FixedTrait::<FP16x16>::new(196607, false)) < ERROR_THRESHOLD, 'mean_X_1 incorrect'); // ~3.0
    assert(FixedTrait::abs(mu_X.get(0, 1).unwrap() - FixedTrait::<FP16x16>::new(458751, false)) < ERROR_THRESHOLD, 'mean_X_2 incorrect'); // ~7.0
    assert(FixedTrait::abs(mu_X.get(0, 2).unwrap() - FixedTrait::<FP16x16>::new(222822, false)) < ERROR_THRESHOLD, 'mean_X_3 incorrect'); // ~3.4
}

#[test]
#[available_gas(200000000000)]
fn covariance_test() {
    let ERROR_THRESHOLD = FixedTrait::<FP16x16>::new_unscaled(100, false); // ~0.00153 error threshold
    
    let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
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

    let mut sigma2_X = MatrixStatisticsTrait::<FP16x16>::covariance(ref X);

    assert(sigma2_X.rows == 3 && sigma2_X.cols == 3, 'Shape incorrect');
    assert(FixedTrait::abs(sigma2_X.get(0, 0).unwrap() - FixedTrait::<FP16x16>::new(109226, false)) < ERROR_THRESHOLD, 'sigma2_X_11 incorrect'); // ~1.67
    assert(FixedTrait::abs(sigma2_X.get(1, 0).unwrap() - FixedTrait::<FP16x16>::new(109226, false)) < ERROR_THRESHOLD, 'sigma2_X_21 incorrect'); // ~1.67
    assert(FixedTrait::abs(sigma2_X.get(2, 0).unwrap() - FixedTrait::<FP16x16>::new(87381, false)) < ERROR_THRESHOLD, 'sigma2_X_31 incorrect'); // ~1.33
    assert(FixedTrait::abs(sigma2_X.get(0, 1).unwrap() - FixedTrait::<FP16x16>::new(109226, false)) < ERROR_THRESHOLD, 'sigma2_X_12 incorrect'); // ~1.67
    assert(FixedTrait::abs(sigma2_X.get(1, 1).unwrap() - FixedTrait::<FP16x16>::new(109226, false)) < ERROR_THRESHOLD, 'sigma2_X_22 incorrect'); // ~1.67
    assert(FixedTrait::abs(sigma2_X.get(2, 1).unwrap() - FixedTrait::<FP16x16>::new(87381, false)) < ERROR_THRESHOLD, 'sigma2_X_32 incorrect'); // ~1.33
    assert(FixedTrait::abs(sigma2_X.get(0, 2).unwrap() - FixedTrait::<FP16x16>::new(87381, false)) < ERROR_THRESHOLD, 'sigma2_X_13 incorrect'); // ~1.33
    assert(FixedTrait::abs(sigma2_X.get(1, 2).unwrap() - FixedTrait::<FP16x16>::new(87381, false)) < ERROR_THRESHOLD, 'sigma2_X_23 incorrect'); // ~1.33
    assert(FixedTrait::abs(sigma2_X.get(2, 2).unwrap() - FixedTrait::<FP16x16>::new(87381, false)) < ERROR_THRESHOLD, 'sigma2_X_33 incorrect'); // ~1.33
}

#[test]
#[available_gas(200000000000)]
fn covariance_weighted_test() {
    let ERROR_THRESHOLD = FixedTrait::<FP16x16>::new_unscaled(100, false); // ~0.00153 error threshold
    
    let mut X = MutMatrixTrait::<FP16x16>::new(4,3);
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

    let mut weights = MutMatrixTrait::<FP16x16>::new(4,1);
    weights.set(0, 0, FixedTrait::<FP16x16>::new(6554, false)); // 0.1
    weights.set(1, 0, FixedTrait::<FP16x16>::new(13107, false)); // 0.2
    weights.set(2, 0, FixedTrait::<FP16x16>::new(19661, false)); // 0.3
    weights.set(3, 0, FixedTrait::<FP16x16>::new(26214, false)); // 0.4

    let mut sigma2_X = MatrixStatisticsTrait::<FP16x16>::covariance_weighted(ref X, ref weights);

    assert(sigma2_X.rows == 3 && sigma2_X.cols == 3, 'Shape incorrect');
    assert(FixedTrait::abs(sigma2_X.get(0, 0).unwrap() - FixedTrait::<FP16x16>::new(93613, false)) < ERROR_THRESHOLD, 'sigma2_X_11 incorrect'); // ~1.43
    assert(FixedTrait::abs(sigma2_X.get(1, 0).unwrap() - FixedTrait::<FP16x16>::new(93613, false)) < ERROR_THRESHOLD, 'sigma2_X_21 incorrect'); // ~1.43
    assert(FixedTrait::abs(sigma2_X.get(2, 0).unwrap() - FixedTrait::<FP16x16>::new(74889, false)) < ERROR_THRESHOLD, 'sigma2_X_31 incorrect'); // ~1.14
    assert(FixedTrait::abs(sigma2_X.get(0, 1).unwrap() - FixedTrait::<FP16x16>::new(93613, false)) < ERROR_THRESHOLD, 'sigma2_X_12 incorrect'); // ~1.43
    assert(FixedTrait::abs(sigma2_X.get(1, 1).unwrap() - FixedTrait::<FP16x16>::new(93613, false)) < ERROR_THRESHOLD, 'sigma2_X_22 incorrect'); // ~1.43
    assert(FixedTrait::abs(sigma2_X.get(2, 1).unwrap() - FixedTrait::<FP16x16>::new(74889, false)) < ERROR_THRESHOLD, 'sigma2_X_32 incorrect'); // ~1.14
    assert(FixedTrait::abs(sigma2_X.get(0, 2).unwrap() - FixedTrait::<FP16x16>::new(74889, false)) < ERROR_THRESHOLD, 'sigma2_X_13 incorrect'); // ~1.14
    assert(FixedTrait::abs(sigma2_X.get(1, 2).unwrap() - FixedTrait::<FP16x16>::new(74889, false)) < ERROR_THRESHOLD, 'sigma2_X_23 incorrect'); // ~1.14
    assert(FixedTrait::abs(sigma2_X.get(2, 2).unwrap() - FixedTrait::<FP16x16>::new(78632, false)) < ERROR_THRESHOLD, 'sigma2_X_33 incorrect'); // ~1.20
}