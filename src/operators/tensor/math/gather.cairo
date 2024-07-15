use core::option::OptionTrait;
use core::traits::TryInto;
use alexandria_data_structures::span_ext::SpanTraitExt;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: TensorTrait::gather docstring
fn gather<T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,>(
    self: @Tensor<T>, indices: Tensor<i32>, axis: Option<i32>
) -> Tensor<T> {
    let axis: usize = match axis {
        Option::Some(val) => {
            if val < 0 {
                (((*self.shape).len()).try_into().unwrap() + val).try_into().unwrap()
            } else {
                val.try_into().unwrap()
            }
        },
        Option::None => 0
    };
    assert(axis < (*self.shape).len(), 'axis out of dimensions');

    let axis_shape = *(*self.shape).at(axis);

    // Adjust indices that are negative
    let mut adjusted_indices = array![];
    let mut indices_data = indices.data.clone();
    loop {
        match indices_data.pop_front() {
            Option::Some(index) => {
                let adjusted_index: usize = if *index < 0 {
                    let val: u32 = (axis_shape.try_into().unwrap() + *index).try_into().unwrap();
                    val
                } else {
                    let val: u32 = (*index).try_into().unwrap();
                    val
                };
                assert(adjusted_index >= 0 && adjusted_index < axis_shape, 'Index out of bounds');
                adjusted_indices.append(adjusted_index);
            },
            Option::None => { break; }
        };
    };

    let mut output_data = array![];
    let mut output_size = array![];
    let mut self_shape = *self.shape;
    let mut i: usize = 0;
    loop {
        match self_shape.pop_front() {
            Option::Some(val) => {
                if i == axis {
                    let mut indices_shape = indices.shape;
                    loop {
                        match indices_shape.pop_front() {
                            Option::Some(item) => { output_size.append(*item); },
                            Option::None => { break; }
                        };
                    };
                } else {
                    output_size.append(*val);
                }

                i += 1;
            },
            Option::None => { break; }
        };
    };

    let mut outer_loop_break = 1;
    let mut divisor = (*self.data).len();

    let mut self_shape = *self.shape;
    let mut i: usize = 0;
    loop {
        match self_shape.pop_front() {
            Option::Some(val) => {
                if i == axis {
                    divisor /= *val;
                    break ();
                };

                outer_loop_break *= *val;
                divisor /= *val;
                i += 1;
            },
            Option::None => { break; }
        };
    };

    let mut break_loop: usize = 1;
    let mut self_shape = *self.shape;
    loop {
        match self_shape.pop_back() {
            Option::Some(val) => {
                if self_shape.len() + 1 == axis {
                    break;
                }
                break_loop *= *val;
            },
            Option::None => { break; }
        };
    };

    let mut outer_loop: usize = 0;
    let axis_index = *self.shape[axis];
    while outer_loop != outer_loop_break {
        let mut adjusted_indices_iter = adjusted_indices.clone();
        loop {
            match adjusted_indices_iter.pop_front() {
                Option::Some(indice) => {
                    let mut inner_loop = 0;
                    while inner_loop != break_loop {
                        let new_val = inner_loop / divisor % axis_index;
                        if indice == new_val {
                            output_data.append(*self.data[break_loop * outer_loop + inner_loop]);
                        }

                        inner_loop += 1;
                    }
                },
                Option::None => { break; },
            };
        };

        outer_loop += 1;
    };

    let mut output_tensor = TensorTrait::<T>::new(output_size.span(), output_data.span());

    output_tensor
}
