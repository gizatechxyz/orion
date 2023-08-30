use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: TensorTrait::asin docstring
fn asin<
    F,
    MAG,
    impl FFixedTrait: FixedTrait<F, MAG>,
    impl FTensor: TensorTrait<F, F>,
    impl FCopy: Copy<F>,
    impl FDrop: Drop<F>,
>(
    mut self: Tensor<F>
) -> Tensor<F> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                result.append((*item).asin());
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(self.shape, result.span(), self.extra);
}
