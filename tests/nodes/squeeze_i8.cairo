mod input_0;
mod output_0;


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::I8Tensor;
use orion::operators::tensor::I8TensorPartialEq;
use orion::numbers::signed_integer::i32::{i32, IntegerTrait};

#[test]
#[available_gas(2000000000)]
fn test_squeeze_i8() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0
        .squeeze(
            Option::Some(array![i32 { mag: 0, sign: false }, i32 { mag: 2, sign: false }].span())
        );

    assert(y.shape == z.shape, 'shapes do not match');
}
