use alexandria_data_structures::array_ext::SpanTraitExt;
use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;

use core::traits::Into;
use core::debug::PrintTrait;
use core::traits::TryInto;
use core::serde::Serde;
use core::traits::Destruct;

use orion::numbers::NumberTrait;
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

/// Cf: TensorTrait::compare docstring
fn compress<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, condition: Tensor<usize>, axis: Option<usize>
) -> Tensor<T> {
    let axis = match axis {
        Option::Some(val) => val,
        Option::None(_) => 999
    };

    let data_rank = (*self.shape).len();
    let condition_rank = (condition.shape).len();
    assert((data_rank >= 1 ), 'data rank must > 1');
    assert((condition_rank == 1), 'condition rank must be 1');

    let mut data_shape = *self.shape;
    let mut condition_shape = condition.shape;

    if (axis != 999) {
        assert(*data_shape.at(axis) >= condition.data.len(), 'index out of bound');
    }

    let mut output_shape = ArrayTrait::new();
    let mut index_data = ArrayTrait::new();
    let mut output_data = ArrayTrait::new();

    let mut data = *self.data;
    let mut condition_data = condition.data;

    let mut ind = 0;
    let mut condition_data_clone = condition_data.clone();
    let mut output = 0;
    loop {
        match condition_data_clone.pop_front() {
            Option::Some(val) => {
                if (*val != 0) {
                    output += 1;
                }
                ind += 1;
            },
            Option::None(_) => { break; }
        };
    };

    if (axis == 999) {
        output_shape.append(output);

        let mut total_shape = 1;
        loop {
            match data_shape.pop_front() {
                Option::Some(val) => {
                    total_shape *= *val;
                },
                Option::None(_) => { break; }
            };
        };

        let mut ind = 0;
        loop {
            match condition_data.pop_front() {
                Option::Some(val) => {
                    if (ind == total_shape) {break; }
                    if (*val != 0){
                        output_data.append(*self.data[ind]);
                    }
                    ind += 1;
                },
                Option::None(_) => { break; }
            };
        };
    } else {
        let mut ind = 0;
        let mut loop_breaker = 1;
        let mut other_loop_breaker = 1;
        let mut multiplier = 1;

        let mut data_shape_clone = data_shape.clone();
        loop {
            match data_shape_clone.pop_front() {
                Option::Some(val) => {
                    if (ind == axis) {
                        output_shape.append(output);
                    }
                    else {
                        output_shape.append(*val);
                        if (ind > axis) {
                            loop_breaker *= *val;
                        }
                        if (ind >= axis) {
                            multiplier *= *val;
                        }
                        if (ind < axis) {
                            other_loop_breaker *= *val;
                        }
                    }
                    ind += 1;
                },
                Option::None(_) => { break; }
            };
        };

        let mut ind = 0;
        let mut ind_loop = 0;
        
        let mut inner_index: usize = 0;
        let mut condition_data_clone = condition_data.clone();

        loop {
            if (ind == other_loop_breaker) {break;}
            let mut condition_data_clone = condition_data.clone();
             inner_index = *data_shape.at(axis) * ind;
            loop {
               
                match condition_data_clone.pop_front() {
                Option::Some(val) => {
                    if (*val != 0){
                        let result = inner_index * loop_breaker ;
                    
                        let mut data_ind:usize = result ;
                        loop {
                            if data_ind == result + loop_breaker { break; }
                            index_data.append(data_ind);
                            data_ind+=1;
                        };
                    }
                inner_index += 1;
                },
                Option::None(_) => { break; }
                };
            };

            ind += 1;
        };

        loop {
            match index_data.pop_front() {
                Option::Some(val) => {
                output_data.append(*self.data[val]);
                },
                Option::None(_) => { break; }
            };
        }; 
    }

    let mut output_tensor = TensorTrait::<T>::new(output_shape.span(), output_data.span());
    return output_tensor;
}