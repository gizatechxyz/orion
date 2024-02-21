use core::array::ArrayTrait;
use core::option::OptionTrait;
use core::array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

use core::dict::Felt252DictTrait;
use core::nullable::{nullable_from_box, match_nullable, FromNullableResult};
use core::debug::PrintTrait;

use core::traits::Into;
use core::traits::TryInto;
/// Cf: TensorTrait::label_encoder docstring
fn label_encoder<
    T,
    +Drop<T>,
    +Copy<T>,
    +AddEq<T>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +Into<T, felt252>,
>(
    // self: @Tensor<T>, default: T, keys: Array<T>, values: Array<T>
    self: @Tensor<T>, default_list: Option<Span<T>>, default_tensor: Option<Tensor<T>>, keys: Option<Span<T>>, keys_tensor: Option<Tensor<T>>, values: Option<Span<T>>, values_tensor: Option<Tensor<T>>,

) -> Tensor<T> 
{
    let mut default = match default_list {
        Option::Some(value) => value,
        Option::None => { 
            match default_tensor {
                Option::Some(value) => value.data,
                Option::None => { core::panic_with_felt252('None') },
            }
        }
    };

    let default = match default.pop_front() {
            Option::Some(value) => *value,
            Option::None => { core::panic_with_felt252('None') }
    };

    let mut keys = match keys {
        Option::Some(value) => { value },
        Option::None => { 
            match keys_tensor {
                Option::Some(value) => { value.data },
                Option::None => { core::panic_with_felt252('None')  },
            }
        }
    };

     let mut values = match values {
        Option::Some(value) => { value },
        Option::None => { 
            match values_tensor {
                Option::Some(value) => { value.data },
                Option::None => { core::panic_with_felt252('None')  },
            }
        }
    };


    assert(keys.len() == values.len(), 'keys must be eq to values');
    let mut key_value_dict: Felt252Dict<Nullable<T>> = Default::default();
    let mut output_data = ArrayTrait::<T>::new();

    loop {
        let key = match keys.pop_front() {
            Option::Some(key) => key,
            Option::None => { break; }
        };
        let value = match values.pop_front() {
            Option::Some(value) => value,
            Option::None => { break; }
        }; 

        key_value_dict.insert((*key).into(), nullable_from_box(BoxTrait::new(*value)));
    };

    let mut data = *self.data;
    loop {
        match data.pop_front() {
            Option::Some(val) => {
                let value = *val;
                let res = key_value_dict.get(value.into());

                let mut span = match match_nullable(res) {
                        FromNullableResult::Null => default,
                        FromNullableResult::NotNull(res) => res.unbox(),
                };
                output_data.append(span);
            },
            Option::None => { break; }
        };
    };

    let mut output_tensor = TensorTrait::<T>::new(*self.shape, output_data.span());
    return output_tensor;
}