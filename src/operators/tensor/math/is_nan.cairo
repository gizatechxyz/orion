use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::U32Tensor;

/// Cf: TensorTrait::is_nan docstring
fn is_nan<
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
                if (*item).is_nan() {
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
