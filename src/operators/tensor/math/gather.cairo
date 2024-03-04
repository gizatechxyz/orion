use alexandria_data_structures::array_ext::SpanTraitExt;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: TensorTrait::gather docstring
fn gather<T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,>(
    self: @Tensor<T>, indices: Tensor<usize>, axis: Option<usize>
) -> Tensor<T> {
    let axis = match axis {
        Option::Some(val) => val,
        Option::None => 0
    };
    assert(axis < (*self.shape).len(), 'axis out of dimensions');

    let axis_shape = *(*self.shape).at(axis);
    let ind_max = indices.data.max().unwrap();
    assert(ind_max < axis_shape, 'this index out of bounds');

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
        let mut data_indices = indices.data;
        loop {
            match data_indices.pop_front() {
                Option::Some(indice) => {
                    let mut inner_loop = 0;
                    while inner_loop != break_loop {
                        let new_val = inner_loop / divisor % axis_index;
                        if *indice == new_val {
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
