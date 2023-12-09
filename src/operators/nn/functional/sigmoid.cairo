use core::traits::Into;
use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;


use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: NNTrait::sigmoid docstring
fn sigmoid<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAdd: Add<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    mut z: Tensor<T>
) -> Tensor<T> {
    let mut data_result = ArrayTrait::<T>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let result = NumberTrait::one()
                    / (NumberTrait::one() + (*item * NumberTrait::neg_one()).exp());
                data_result.append(result);
            },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(z.shape, data_result.span());
}

