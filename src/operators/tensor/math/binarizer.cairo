use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: TensorTrait::binarizer docstring
fn binarizer<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut self: Tensor<T>, threshold: Option<T>
) -> Tensor<T> {
    let threshold: T = if threshold.is_some() {
        threshold.unwrap()
    } else {
        NumberTrait::zero()
    };

    let mut binarized_data: Array<T> = array![];

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                if (*item) > threshold {
                    binarized_data.append(NumberTrait::one());
                } else {
                    binarized_data.append(NumberTrait::zero());
                }
            },
            Option::None => { break; }
        };
    };

    TensorTrait::new(self.shape, binarized_data.span())
}
