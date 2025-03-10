use alexandria_data_structures::span_ext::SpanTraitExt;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: TensorTrait::gather_elements docstring
fn gather_elements<T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,>(
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
    let mut data_shape_clone = (*self.shape).clone();
    let mut multiplier = 1;
    let mut looper = 1;
    let mut ind = 0;
    loop {
        match data_shape_clone.pop_front() {
            Option::Some(val) => {
                if ind >= axis {
                    multiplier *= *val;
                }
                if ind > axis {
                    looper *= *val;
                }
                ind += 1;
            },
            Option::None => { break; }
        };
    };

    let inner_loop = multiplier / axis_shape;
    let mut adjusted_indices_iter = adjusted_indices.clone();

    let mut i: usize = 0;
    loop {
        match adjusted_indices_iter.pop_front() {
            Option::Some(indice) => {
                let value = if axis == 0 {
                    indice * inner_loop + (i % inner_loop)
                } else if axis == (*self.shape).len() - 1 {
                    indice + axis_shape * (i / axis_shape)
                } else {
                    indice * looper
                        + (i % looper)
                        + (multiplier / axis_shape) * (i / (multiplier / axis_shape))
                };

                output_data.append(*self.data[value]);
                i += 1;
            },
            Option::None => { break; }
        };
    };

    TensorTrait::<T>::new(indices.shape, output_data.span())
}
