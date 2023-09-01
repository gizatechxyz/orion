use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: TensorTrait::ceil docstring
fn ceil<
    F,
    MAG,
    impl FFixedTrait: FixedTrait<F, MAG>,
    impl FTensor: TensorTrait<F, F>,
    impl FCopy: Copy<F>,
    impl FDrop: Drop<F>
>(
    mut z: Tensor<F>
) -> Tensor<F> {
    let mut data_result = ArrayTrait::<F>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                data_result.append((*item).ceil());
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span());
}

