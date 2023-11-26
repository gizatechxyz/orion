mod input_0;
mod output_0;
mod output_1;
mod output_2;
mod output_3;


use orion::operators::tensor::I32TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::U32TensorPartialEq;

use debug::PrintTrait;

#[test]
#[available_gas(2000000000)]
fn test_unique_u32_with_axis_one_not_sorted() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();
    let z_1 = output_1::output_1();
    let z_2 = output_2::output_2();
    let z_3 = output_3::output_3();

    let (y_0, y_1, y_2, y_3) = input_0.unique(Option::Some(1), Option::Some(false));
    
    let mut x = y_0.data;
    let mut i = 0;
    loop {
        match x.pop_front() {
            Option::Some(v) => {
                let m: felt252 = (*v).into();
                if i == 9 {
                    '------'.print();
                    i = 0;
                }
                m.print();
            },
            Option::None => {
                break;
            },
        }
        i += 1;
    };
    
    'a'.print();
    assert_eq(y_3, z_3);
    'b'.print();
    assert_eq(y_1, z_1);
    'c'.print();
    assert_eq(y_2, z_2);
    'd'.print();
    assert_eq(y_0, z_0);
    'e'.print();
}
