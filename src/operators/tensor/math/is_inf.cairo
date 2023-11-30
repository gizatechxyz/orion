use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::tensor_bool::BoolTensor;

/// Cf: TensorTrait::is_inf docstring
fn is_inf<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(x: @Tensor<T>, detect_negative: Option<u8>, detect_positive: Option<u8>) -> Tensor<bool> {
    let neg_opt = match detect_negative {
        Option::Some(val) => {
	    if val == 0 { 0 } else { 1 }
	},
	Option::None => 1,
    };

    let pos_opt = match detect_positive {
        Option::Some(val) => {
	    if val == 0 { 0 } else { 1 }
	},
	Option::None => 1,
    };

    if neg_opt == 0 && pos_opt == 0 {
	return TensorTrait::new(*x.shape, ArrayTrait::<bool>::new().span());
    }

    if neg_opt == 0 && pos_opt == 1 {
        return is_pos_inf(x);
    }

    if neg_opt == 1 && pos_opt == 0 {
	return is_neg_inf(x);
    }

    let mut data_result = ArrayTrait::<bool>::new();
    let mut y: Span<T> = *x.data;
    loop {
        match y.pop_front() {
            Option::Some(item) => {
    	        data_result.append((*item).is_inf());
    	    },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(*x.shape, data_result.span());
}

/// Cf: TensorTrait::is_pos_inf docstring
fn is_pos_inf<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(x: @Tensor<T>) -> Tensor<bool> {
    let mut data_result = ArrayTrait::<bool>::new();
    let mut y: Span<T> = *x.data;
    loop {
        match y.pop_front() {
            Option::Some(item) => {
	    	data_result.append((*item).is_pos_inf());
	    },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(*x.shape, data_result.span());
}

/// Cf: TensorTrait::is_neg_inf docstring
fn is_neg_inf<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(x: @Tensor<T>) -> Tensor<bool> {
    let mut data_result = ArrayTrait::<bool>::new();
    let mut y: Span<T> = *x.data;
    loop {
        match y.pop_front() {
            Option::Some(item) => {
	    	data_result.append((*item).is_neg_inf());
	    },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(*x.shape, data_result.span());
}
