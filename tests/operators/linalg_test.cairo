use core::array::{ArrayTrait, SpanTrait};
use core::debug::PrintTrait;
use core::option::OptionTrait;
use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::implementations::fp16x16::math::core::{abs};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl};
use orion::operators::matrix::{MutMatrix, MutMatrixTrait, MutMatrixImpl};
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};
use orion::operators::matrix::{MutMatrix, MutMatrixTrait, MutMatrixImpl};

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

#[test]
#[available_gas(99999999999999999)]
fn linalg_test() {
    let mut X = MutMatrixTrait::<FP16x16>::new(rows: 3, cols: 3);
    X.set(0, 0, FP16x16 { mag: 131072, sign: false }); // 2
    X.set(0, 1, FP16x16 { mag: 65536, sign: false }); // 1
    X.set(0, 2, FP16x16 { mag: 65536, sign: true }); // -1
    X.set(1, 0, FP16x16 { mag: 196608, sign: true }); // -3
    X.set(1, 1, FP16x16 { mag: 65536, sign: true }); // -1
    X.set(1, 2, FP16x16 { mag: 131072, sign: false }); // 2
    X.set(2, 0, FP16x16 { mag: 131072, sign: true }); // -2
    X.set(2, 1, FP16x16 { mag: 65536, sign: false }); // 1
    X.set(2, 2, FP16x16 { mag: 131072, sign: false }); // 1

    let mut y = MutMatrixTrait::<FP16x16>::new(rows: 3, cols: 1);
    y.set(0, 0, FP16x16 { mag: 524288, sign: false }); // 8
    y.set(0, 1, FP16x16 { mag: 720896, sign: true }); // -11
    y.set(0, 2, FP16x16 { mag: 196608, sign: true }); // -3
    
    let mut S = linalg_solve(ref X, ref y);
    // S = [2, 3, -1]

    assert(S.rows == X.rows, 'Wrong num of rows');
    assert(S.cols == 1, 'Wrong num of cols');

    // Check mags
    let threshold = FP16x16 {mag: 100, sign: false}; // ~0.00153 error threshold
    assert(abs(S.get(0, 0).unwrap() - FP16x16 {mag: 131072, sign: false}) <= threshold , 'S_0 mag is wrong');
    assert(abs(S.get(1, 0).unwrap() - FP16x16 {mag: 196608, sign: false}) <= threshold, 'S_1 mag is wrong');
    assert(abs(S.get(2, 0).unwrap() - FP16x16 {mag: 65536, sign: true}) <= threshold, 'S_2 mag is wrong');
    
    // Check signs
    assert(S.get(0, 0).unwrap().sign == false, 'S_1 sign is wrong');
    assert(S.get(1, 0).unwrap().sign == false, 'S_2 sign is wrong');
    assert(S.get(2, 0).unwrap().sign == true, 'S_3 sign is wrong');

}