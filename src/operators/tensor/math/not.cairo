use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::{tensor_bool::BoolTensor};


// Cf TensorTrait::not docstring
fn not <
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
> (mut z: Tensor<bool>) -> Tensor<bool> {
    let mut data_result = ArrayTrait::<bool>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                data_result.append((!*item));
               
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span());
}