use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: NNTrait::thresholded_relu docstring
fn thresholded_relu<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut z: Tensor<T>, alpha: @T
) -> Tensor<T> {
    let mut data_result: Array<T> = array![];

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                if (*item) <= (*alpha) {
                    data_result.append(NumberTrait::zero());
                } else {
                    data_result.append(*item);
                };
            },
            Option::None => { break; }
        };
    };

    TensorTrait::new(z.shape, data_result.span())
}
