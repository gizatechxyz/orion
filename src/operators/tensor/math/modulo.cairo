use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;
use core::array::{ArrayTrait, SpanTrait};
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::{FP16x16, FP16x16Impl};

use orion::operators::tensor::core::{stride};
use orion::operators::tensor::{FP16x16Tensor, U32Tensor,};
use orion::operators::vec::{NullableVec, NullableVecImpl};

/// Cf: TensorTrait::Mod docstring
fn modulo<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Add<Tensor<T>>,
    +Sub<Tensor<T>>,
    +Div<Tensor<T>>,
    +Mul<Tensor<T>>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +Rem<T>,
>(self: @Tensor<T>, divisor: @Tensor<T>, fmod: Option<bool>) -> Tensor<T> {
    
    let mut dividend = self.clone();
    let mut divisor = divisor.clone();

    match fmod {
        Option::Some(item) => {
            if item == true {
                dividend = self.abs();
                divisor = divisor.abs();
            }
        },
        Option::None => {},
    }

    let mut quotient = dividend / divisor;

    let mut res_data : Array<T> = array![];
    loop {
        match quotient.data.pop_front() {
            Option::Some(val) => {
                if *val % NumberTrait::<T>::one() != NumberTrait::<T>::zero() {
                    let mut temp = NumberTrait::floor(*val);
                    res_data.append(temp);
                } else {
                    res_data.append(*val);
                }
            },
            Option::None => {
                break;
            }
        }
    };

    quotient = TensorTrait::<T>::new(*self.shape, res_data.span());

    let mut remainder = dividend - quotient * divisor;

    if fmod.is_some() && fmod.unwrap() == true {
        remainder = remainder * self.sign();
    }  
    
    return remainder;
}

