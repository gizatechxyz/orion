use core::traits::Into;
use core::traits::TryInto;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use core::option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;

use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};
use core::traits::PartialEq;
use alexandria_merkle_tree::merkle_tree::{pedersen::PedersenHasherImpl};
use core::integer::{u128s_from_felt252, U128sFromFelt252Result};
use core::traits;

/// Cf: TensorTrait::random_uniform_like docstring
fn random_uniform_like<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TRem: Rem<T>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    tensor: Tensor<T>, high: Option<T>, low: Option<T>, seed: Option<usize>
) -> Tensor<T> {
    let mut seed: usize = match seed {
        Option::Some(seed) => seed,
        Option::None => NumberTrait::max_value(),
    };
    let mut high = match high {
        Option::Some(high) => high,
        Option::None => NumberTrait::one(),
    };
    let mut low = match low {
        Option::Some(low) => low,
        Option::None => NumberTrait::zero(),
    };
    assert!(high > low, "high must be larger than low");
    let res = tensor_get_state(tensor, seed, high, low);

    return res;
}


fn tensor_get_state<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TRem: Rem<T>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    tensor: Tensor<T>, mut seed: usize, high: T, low: T
) -> Tensor<T> {
    let mut data = ArrayTrait::new();
    let mut count = (tensor.data).len();
    let mut i = 0;

    loop {
        if count == i {
            break;
        }
        let mut v = NumberTrait::one();
        v = hash_random_range(seed, low, high);
        let a: u64 = 1664525;
        let c: u64 = 1013904223;
        let m: u64 = 4294967295;
        let s: u64 = (a * seed.try_into().unwrap() + c) % m;
        seed = s.try_into().unwrap();
        data.append(v);
        i += 1;
    };
    return TensorTrait::new(tensor.shape, data.span());
}

// High level random in a range
// Only one random number per hash might be inefficient.
fn hash_random_range<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TRem: Rem<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    seed: usize, min: T, max: T
) -> T {
    let mut key = PedersenHasherImpl::new();
    let hash: felt252 = key.hash(seed.into(), 1);
    let a: u128 = 4294967295;
    let b: u128 = match u128s_from_felt252(hash) {
        U128sFromFelt252Result::Narrow(x) => x,
        U128sFromFelt252Result::Wide((x, _)) => x,
    } % a;
    let c: felt252 = b.into();
    let rnd: T = NumberTrait::from_felt(c);
    let range = max - min + NumberTrait::one(); // + 1 to include max
    min + rnd % range
}
