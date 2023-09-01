use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: TensorTrait::acos docstring
fn acos<
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
                result.append((*item).acos());
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<F>::new(self.shape, result.span());
}

