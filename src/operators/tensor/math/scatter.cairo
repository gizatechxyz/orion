use alexandria_data_structures::array_ext::ArrayTraitExt;
use alexandria_data_structures::span_ext::SpanTraitExt;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use core::dict::Felt252DictTrait;
use core::nullable::{nullable_from_box, match_nullable, FromNullableResult};

/// Cf: TensorTrait::scatter docstring
fn scatter<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TAddEq: AddEq<T>,
    impl TMulEq: MulEq<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
>(
    self: @Tensor<T>,
    updates: Tensor<T>,
    indices: Tensor<usize>,
    axis: Option<usize>,
    reduction: Option<usize>
) -> Tensor<T> {
    let mut axis = match axis {
        Option::Some(val) => val,
        Option::None => 0
    };

    let reduction = match reduction {
        Option::Some(val) => val,
        Option::None => 'none'
    };

    let data_rank = (*self.shape).len();
    let indices_rank = (indices.shape).len();
    let updates_rank = (updates.shape).len();

    assert((data_rank == updates_rank) & (updates_rank == indices_rank), 'must be same rank');
    let data_shape = *self.shape;
    let mut indices_arr: Array<usize> = array![];
    indices_arr.extend_from_span(indices.data);
    let ind_max = indices_arr.max().unwrap();
    assert(ind_max < *data_shape.at(axis), 'index is out of bound');

    let data_shape = *self.shape;
    let mut indices_shape = indices.shape;
    let updates_shape = updates.shape;
    assert(
        (*indices_shape[0] == *updates_shape[0]) & (*indices_shape[1] == *updates_shape[1]),
        'shape must be same'
    );

    let mut output_data = array![];
    let mut data_indices = indices.data;
    let mut data_updates = updates.data;
    let mut indices_updates: Felt252Dict<usize> = Default::default();
    let mut indices_updates_reduction: Felt252Dict<Nullable<Span<usize>>> = Default::default();

    let mut data_shape_copy = data_shape;
    let mut indices_shape_copy = indices_shape;
    *data_shape_copy.pop_front().unwrap();
    *indices_shape_copy.pop_front().unwrap();

    let mut indices_loop: usize = 1;
    let mut data_loop: usize = 1;

    if (axis == 0) {
        loop {
            match indices_shape_copy.pop_front() {
                Option::Some(val) => { indices_loop *= *val; },
                Option::None => { break; }
            };
        };

        loop {
            match data_shape_copy.pop_front() {
                Option::Some(val) => { data_loop *= *val; },
                Option::None => { break; }
            };
        };
    }

    let mut transpose = false;
    if ((data_rank > 2) & (axis == 1)) {
        let index = indices.transpose(axes: array![0, 2, 1].span());
        let update = updates.transpose(axes: array![0, 2, 1].span());
        data_indices = index.data;
        data_updates = update.data;
        indices_shape = index.shape;
        axis = 2;
        transpose = true;
    }

    if (axis == (data_rank - 1)) {
        data_loop = *data_shape_copy.pop_back().unwrap();
        indices_loop = *indices_shape_copy.pop_back().unwrap();
    }

    let mut total_count: usize = 0;
    let mut shift = 0;

    loop {
        let mut result: usize = 0;

        match data_indices.pop_front() {
            Option::Some(val) => {
                let value = total_count + 1;

                if (axis == 0) {
                    let column = total_count % indices_loop;
                    result = (*val * data_loop) + (column);
                    if ((result % *data_shape.at(data_rank - 1)) != total_count % *indices_shape
                        .at(data_rank - 1)) {
                        result +=
                            (*data_shape.at(data_rank - 1) - *indices_shape.at(data_rank - 1));
                    }
                }

                if (axis == (data_rank - 1)) {
                    let mut row = total_count / indices_loop;
                    if ((data_rank > 2) & (row % *data_shape.at(1) >= *indices_shape.at(1))) {
                        shift = (*data_shape.at(1) - *indices_shape.at(1));
                    }

                    result = *val + (data_loop * (row + shift));
                }

                if (reduction == 'none') {
                    indices_updates.insert(result.into(), value.into());
                } else {
                    let mut arr = array![];

                    let val = indices_updates_reduction.get(result.into());
                    let mut a = ArrayTrait::new();
                    let mut span = match match_nullable(val) {
                        FromNullableResult::Null(()) => a.span(),
                        FromNullableResult::NotNull(val) => val.unbox(),
                    };

                    loop {
                        match span.pop_front() {
                            Option::Some(val) => { arr.append(*val); },
                            Option::None => { break; }
                        };
                    };

                    arr.append(total_count);
                    indices_updates_reduction
                        .insert(result.into(), nullable_from_box(BoxTrait::new(arr.span())));
                }

                total_count += 1;
            },
            Option::None => { break; }
        };
    };

    let mut data = *self.data;
    let mut i: usize = 0;
    loop {
        match data.pop_front() {
            Option::Some(val) => {
                if (reduction == 'none') {
                    let value = indices_updates.get(i.into());
                    if (value == 0) {
                        output_data.append(*val);
                    } else {
                        let data_value = data_updates[value - 1];
                        output_data.append(*data_value);
                    }
                } else {
                    let value = indices_updates_reduction.get(i.into());
                    let mut a = array![];
                    let mut span = match match_nullable(value) {
                        FromNullableResult::Null(()) => a.span(),
                        FromNullableResult::NotNull(value) => value.unbox(),
                    };

                    if (span.len() == 0) {
                        output_data.append(*val);
                    } else {
                        // let mut result = *data_updates.at(*span.pop_front().unwrap());
                        let mut result = *val;

                        if (reduction == 'add') {
                            loop {
                                match span.pop_front() {
                                    Option::Some(val) => { result += *data_updates[*val]; },
                                    Option::None => { break; }
                                };
                            };

                            output_data.append(result);
                        }

                        if (reduction == 'mul') {
                            loop {
                                match span.pop_front() {
                                    Option::Some(val) => { result *= *data_updates[*val]; },
                                    Option::None => { break; }
                                };
                            };

                            output_data.append(result);
                        }

                        if (reduction == 'max') {
                            loop {
                                match span.pop_front() {
                                    Option::Some(val) => {
                                        let holder = *data_updates[*val];
                                        if (holder > result) {
                                            result = holder;
                                        }
                                    },
                                    Option::None => { break; }
                                };
                            };

                            output_data.append(result);
                        }

                        if (reduction == 'min') {
                            loop {
                                match span.pop_front() {
                                    Option::Some(val) => {
                                        let holder = *data_updates[*val];
                                        if (holder < result) {
                                            result = holder;
                                        }
                                    },
                                    Option::None => { break; }
                                };
                            };

                            output_data.append(result);
                        }
                    }
                }

                i += 1;
            },
            Option::None => { break; }
        };
    };

    let mut output_tensor = TensorTrait::<T>::new(*self.shape, output_data.span());

    if transpose {
        output_tensor = output_tensor.transpose(axes: array![0, 2, 1].span())
    }

    output_tensor
}
