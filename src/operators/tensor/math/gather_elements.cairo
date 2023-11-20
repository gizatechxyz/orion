use alexandria_data_structures::array_ext::SpanTraitExt;
use array::ArrayTrait;
use array::SpanTrait;

use core::traits::Into;
use debug::PrintTrait;
use core::traits::TryInto;
use core::serde::Serde;
use core::traits::Destruct;
use option::OptionTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

/// Cf: TensorTrait::gather docstring
fn gather_elements<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, indices: Tensor<usize>, axis: Option<usize>
) -> Tensor<T> {
    let axis = match axis {
        Option::Some(val) => val,
        Option::None(_) => 0
    };
    assert(axis < (*self.shape).len(), 'axis out of dimensions');

    let data_rank = (*self.shape).len();
    let indices_rank = (indices.shape).len();
    assert((data_rank == indices_rank ) & (indices_rank >= 1), 'must be same rank');

    let axis_shape = *(*self.shape).at(axis);
    let ind_max = indices.data.max().unwrap();
    assert(ind_max < axis_shape, 'this index out of bounds');

    let mut indices_shape = indices.shape;
    let mut data_shape = *self.shape;
    let mut data_shape_clone = data_shape.clone();


    let mut ind = 0;
    loop {
        match data_shape.pop_front() {
            Option::Some(val) => {
                if (ind != axis) {
                    assert(*val == *indices_shape.at(ind), 'shape mismatch');
                }
                ind += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let mut output_data = ArrayTrait::new();

    let mut outer_loop = data_shape_clone.at(axis);
    let mut inner_loop = 1;
    let mut multiplier = 1;
    let mut ind = 0;
    loop {
        match data_shape_clone.pop_front() {
            Option::Some(val) => {
                inner_loop *= *val;
                if (ind >= axis) {
                    multiplier *= *val;
                }
                ind += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let looper = multiplier / *outer_loop;

    if inner_loop != 1 {
        inner_loop /= *outer_loop;
    }

    let mut multiplier_index = 1;
    let mut outer_loop_index = indices_shape.at(axis);
    let mut ind = 0;
    loop {
        match indices_shape.pop_front() {
            Option::Some(val) => {
                if (ind >= axis) {
                    multiplier_index *= *val;
                }
                ind += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let mut data_indices = indices.data;
    let mut i: usize = 0;
    loop {
        match data_indices.pop_front() {
            Option::Some(val) => {
                if (axis == 0){
                    let value  = *val * inner_loop.into() + (i % inner_loop);
                    output_data.append(*self.data[value]);
                }
                if ((axis == indices_rank-1) & (axis != 0)) {
                    let value = *val + *outer_loop * (i / *outer_loop_index);
                    output_data.append(*self.data[value]);
                }
                if ((axis != indices_rank-1) & (axis != 0)) {
                    let value = *val * (looper ) + (i % looper) + (multiplier  * (i / multiplier_index));
                    output_data.append(*self.data[value]);
                }
                i += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let mut output_tensor = TensorTrait::<T>::new(indices.shape, output_data.span());
    return output_tensor;
}