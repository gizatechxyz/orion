use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;
use core::traits::Into;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};


fn round<
    T,
    MAG,
    impl TNumberTrait: NumberTrait<T, MAG>,
    impl FTensor: TensorTrait<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>
>(
    mut self: Tensor<T>
) -> Tensor<T> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => { result.append((*item).round()); },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(self.shape, result.span());
}
