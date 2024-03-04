use orion::operators::ml::normalizer::normalizer::{NormalizerTrait, NORM};
use orion::utils::{assert_eq, assert_seq_eq};

use orion::numbers::FP16x16;
use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorDiv, FP16x16TensorPartialEq
};

#[test]
#[available_gas(200000000000)]
fn test_normalizer_max() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 52428, sign: true });
    data.append(FP16x16 { mag: 39321, sign: true });
    data.append(FP16x16 { mag: 26214, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 13107, sign: false });
    data.append(FP16x16 { mag: 26214, sign: false });
    data.append(FP16x16 { mag: 39321, sign: false });
    let X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 52428, sign: true });
    data.append(FP16x16 { mag: 39321, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 32768, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 21845, sign: false });
    data.append(FP16x16 { mag: 43690, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    let expected_output = TensorTrait::new(shape.span(), data.span());

    let actual_output = NormalizerTrait::predict(X, NORM::MAX);

    assert_eq(actual_output, expected_output);
}

#[test]
#[available_gas(200000000000)]
fn test_normalizer_l1() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 52428, sign: true });
    data.append(FP16x16 { mag: 39321, sign: true });
    data.append(FP16x16 { mag: 26214, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 13107, sign: false });
    data.append(FP16x16 { mag: 26214, sign: false });
    data.append(FP16x16 { mag: 39321, sign: false });
    let X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 27306, sign: true });
    data.append(FP16x16 { mag: 21845, sign: true });
    data.append(FP16x16 { mag: 16384, sign: true });
    data.append(FP16x16 { mag: 43690, sign: true });
    data.append(FP16x16 { mag: 21845, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 10922, sign: false });
    data.append(FP16x16 { mag: 21845, sign: false });
    data.append(FP16x16 { mag: 32768, sign: false });
    let expected_output = TensorTrait::new(shape.span(), data.span());

    let actual_output = NormalizerTrait::predict(X, NORM::L1);

    assert_eq(actual_output, expected_output);
}

#[test]
#[available_gas(200000000000)]
fn test_normalizer_l2() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 52428, sign: true });
    data.append(FP16x16 { mag: 39321, sign: true });
    data.append(FP16x16 { mag: 26214, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 13107, sign: false });
    data.append(FP16x16 { mag: 26214, sign: false });
    data.append(FP16x16 { mag: 39321, sign: false });
    let X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 46340, sign: true });
    data.append(FP16x16 { mag: 37072, sign: true });
    data.append(FP16x16 { mag: 27804, sign: true });
    data.append(FP16x16 { mag: 58617, sign: true });
    data.append(FP16x16 { mag: 29308, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 17515, sign: false });
    data.append(FP16x16 { mag: 35030, sign: false });
    data.append(FP16x16 { mag: 52545, sign: false });
    let expected_output = TensorTrait::new(shape.span(), data.span());

    let actual_output = NormalizerTrait::predict(X, NORM::L2);

    assert_eq(actual_output, expected_output);
}


#[test]
#[available_gas(200000000000)]
fn test_normalizer_max_avoid_div_zero() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    let X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    let expected_output = TensorTrait::new(shape.span(), data.span());

    let actual_output = NormalizerTrait::predict(X, NORM::MAX);

    assert_eq(actual_output, expected_output);
}

