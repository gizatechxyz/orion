use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: TensorTrait::acos docstring
fn acos<
    T,
    MAG,
    impl FFixedTrait: FixedTrait<T, MAG>,
    impl FTensor: TensorTrait<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    mut self: Tensor<T>
) -> Tensor<T> {
    let mut result = ArrayTrait::new();
    loop {
        match self.data.pop_front() {
            Option::Some(item) => { result.append((*item).acos()); },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::<T>::new(self.shape, result.span());
}

