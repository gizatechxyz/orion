use core::nullable::{nullable_from_box, match_nullable, FromNullableResult};

use alexandria_data_structures::array_ext::ArrayTraitExt;
use alexandria_data_structures::span_ext::SpanTraitExt;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

/// Cf: TensorTrait::scatter_nd docstring
fn scatter_nd<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TAdd: Add<T>,
    impl TMul: Mul<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
>(
    self: @Tensor<T>, updates: Tensor<T>, indices: Tensor<usize>, reduction: Option<usize>
) -> Tensor<T> {
    let reduction = match reduction {
        Option::Some(val) => val,
        Option::None => 'none'
    };

    let data_rank = (*self.shape).len();
    let mut data_shape = *self.shape;
    let mut indices_shape = indices.shape;
    let updates_shape = updates.shape;

    let indices_last_axis = indices_shape.pop_back().unwrap();
    assert(*indices_last_axis <= data_rank, 'must be <= data rank');

    let mut indices_arr: Array<usize> = array![];
    indices_arr.extend_from_span(indices.data);
    let ind_max = indices_arr.max().unwrap();
    if (data_rank > 1) {
        assert(ind_max < data_rank, 'index is out of bound');
    }

    let mut batch_dims_shape = array![];
    let mut ind: usize = 0;

    loop {
        match indices_shape.pop_front() {
            Option::Some(val) => { batch_dims_shape.append(*val); },
            Option::None => { break; }
        };
    };

    let mut data_shape_clone = data_shape.clone();
    loop {
        match data_shape_clone.pop_front() {
            Option::Some(val) => {
                if (ind >= *indices_last_axis) {
                    batch_dims_shape.append(*val);
                }
            },
            Option::None => { break; }
        };
    };

    let mut ind: usize = 0;
    loop {
        match batch_dims_shape.pop_front() {
            Option::Some(val) => { assert(val == *updates_shape[ind], 'must be same'); },
            Option::None => { break; }
        };
    };

    let mut data_indices = indices.data;
    let mut data_updates = updates.data;

    let mut data_shape_clone = data_shape.clone();
    let mut indexer = 1;
    let data_shape_first = data_shape_clone.pop_front();
    if data_rank >= 1 {
        loop {
            match data_shape_clone.pop_front() {
                Option::Some(val) => { indexer *= *val; },
                Option::None => { break; }
            };
        }
    }

    let mut updates_index_dict: Felt252Dict<u32> = Default::default();
    let mut dict_ind: usize = 1;
    loop {
        match data_indices.pop_front() {
            Option::Some(val) => {
                updates_index_dict.insert((*val).into(), dict_ind);
                dict_ind += 1;
            },
            Option::None => { break; }
        };
    };

    let mut output_data: Array<T> = array![];
    let mut data = *self.data;
    let mut index: usize = 0;
    let mut inner_index: usize = 0;
    let num = *data_shape_first.unwrap();
    while index != num {
        let comp_index = updates_index_dict.get(index.into());

        if comp_index == 0 {
            loop {
                if (inner_index == indexer) {
                    inner_index = 0;
                    break;
                }
                let val = *data.at((index * indexer) + inner_index);
                output_data.append(val);
                inner_index += 1;
            };
        } else {
            loop {
                if (inner_index == indexer) {
                    inner_index = 0;
                    break;
                }
                if (reduction == 'none') {
                    let val = data_updates.at(((comp_index - 1) * indexer) + inner_index);
                    output_data.append(*val);
                }
                if (reduction == 'add') {
                    let val = data_updates.at(((comp_index - 1) * indexer) + inner_index);
                    let data_val = *data.at((index * indexer) + inner_index);
                    output_data.append(*val + data_val);
                }
                if (reduction == 'mul') {
                    let val = data_updates.at(((comp_index - 1) * indexer) + inner_index);
                    let data_val = *data.at((index * indexer) + inner_index);
                    output_data.append((*val) * data_val);
                }
                if (reduction == 'max') {
                    let val = data_updates.at(((comp_index - 1) * indexer) + inner_index);
                    let data_val = *data.at((index * indexer) + inner_index);
                    if (*val > data_val) {
                        output_data.append(*val);
                    } else {
                        output_data.append(data_val);
                    }
                }
                if (reduction == 'min') {
                    let val = data_updates.at(((comp_index - 1) * indexer) + inner_index);
                    let data_val = *data.at((index * indexer) + inner_index);
                    if (*val > data_val) {
                        output_data.append(data_val);
                    } else {
                        output_data.append(*val);
                    }
                }
                inner_index += 1;
            }
        }
        index += 1;
    };

    let mut output_tensor = TensorTrait::<T>::new(*self.shape, output_data.span());

    output_tensor
}
