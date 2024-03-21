use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::U32Tensor;

/// Cf: TensorTrait::is_inf docstring
fn is_inf<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    x: @Tensor<T>, detect_negative: Option<u8>, detect_positive: Option<u8>
) -> Tensor<usize> {
    let neg_opt = match detect_negative {
        Option::Some(val) => { if val == 0 {
            0
        } else {
            1
        } },
        Option::None => 1,
    };

    let pos_opt = match detect_positive {
        Option::Some(val) => { if val == 0 {
            0
        } else {
            1
        } },
        Option::None => 1,
    };

    if neg_opt == 0 && pos_opt == 0 {
        return TensorTrait::new(*x.shape, ArrayTrait::<usize>::new().span());
    }

    if neg_opt == 0 && pos_opt == 1 {
        return is_pos_inf(x);
    }

    if neg_opt == 1 && pos_opt == 0 {
        return is_neg_inf(x);
    }

    let mut data_result: Array<usize> = array![];
    let mut y: Span<T> = *x.data;
    loop {
        match y.pop_front() {
            Option::Some(item) => {
                if (*item).is_inf() {
                    data_result.append(1);
                } else {
                    data_result.append(0);
                }
            },
            Option::None => { break; }
        };
    };

    TensorTrait::new(*x.shape, data_result.span())
}

/// Cf: TensorTrait::is_pos_inf docstring
fn is_pos_inf<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    x: @Tensor<T>
) -> Tensor<usize> {
    let mut data_result: Array<usize> = array![];
    let mut y: Span<T> = *x.data;
    loop {
        match y.pop_front() {
            Option::Some(item) => {
                if (*item).is_pos_inf() {
                    data_result.append(1);
                } else {
                    data_result.append(0);
                }
            },
            Option::None => { break; }
        };
    };

    TensorTrait::new(*x.shape, data_result.span())
}

/// Cf: TensorTrait::is_neg_inf docstring
fn is_neg_inf<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    x: @Tensor<T>
) -> Tensor<usize> {
    let mut data_result: Array<usize> = array![];
    let mut y: Span<T> = *x.data;
    loop {
        match y.pop_front() {
            Option::Some(item) => {
                if (*item).is_neg_inf() {
                    data_result.append(1);
                } else {
                    data_result.append(0);
                }
            },
            Option::None => { break; }
        };
    };

    TensorTrait::new(*x.shape, data_result.span())
}
