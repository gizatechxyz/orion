use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: NNTrait::leaky_relu docstring
fn leaky_relu<
    T,
    MAG,
    impl FNumber: NumberTrait<T, MAG>,
    impl FTensor: TensorTrait<T>,
    impl FPartialOrd: PartialOrd<T>,
    impl FMul: Mul<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    mut z: Tensor<T>, alpha: @T
) -> Tensor<T> {
    assert(*alpha < NumberTrait::one(), 'alpha must be less than 1');

    let mut data_result: Array<T> = array![];

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                if (*item >= NumberTrait::zero()) {
                    data_result.append(*item);
                } else {
                    data_result.append(*item * *alpha);
                };
            },
            Option::None => { break; }
        };
    };

    TensorTrait::new(z.shape, data_result.span())
}
