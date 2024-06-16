// use core::array::ToSpanTrait;
use core::array::ArrayTrait;
use orion_cairo::tensors::Tensor;
use orion_cairo::numbers::f16x16::{f16x16, FixedTrait};
use orion_cairo::helpers::{
    broadcast_index_mapping, unravel_index, ravel_index, combine_indices, get_all_axes,
    broadcast_shape, len_from_shape, reduce_output_shape, accumulate_sum, unique, bubble_sort
};


#[generate_trait]
pub impl PrimopsImpl of PrimopsTrait {
    fn mul(self: @Tensor, other: @Tensor, output_shape: Span<usize>) -> Tensor {
        let mut result = array![];

        let num_elements = len_from_shape(output_shape);

        let mut n: usize = 0;
        while n != num_elements {
            let indices_broadcasted = unravel_index(n, output_shape);

            let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
            let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);
            result
                .append(
                    FixedTrait::mul(*(*self.data)[indices_self], *(*other.data)[indices_other])
                );

            n += 1;
        };

        Tensor { shape: output_shape, data: result.span() }
    }
    fn reduce_sum(
        self: @Tensor,
        axes: Option<Span<i32>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor {
        let noop_with_empty_axes = match noop_with_empty_axes {
            Option::Some(noop_with_empty_axes) => noop_with_empty_axes,
            Option::None => false,
        };
        let axes = match axes {
            Option::Some(axes) => {
                if (axes.len() == 0) {
                    get_all_axes(*self.shape)
                } else {
                    assert(axes.len() == unique(axes).len(), 'duplicated axis.');
                    let mut axes_arr: Array<usize> = array![];
                    let mut copy_axes = axes.clone();
                    loop {
                        match copy_axes.pop_front() {
                            Option::Some(axis) => {
                                // Adjust negative axes to positive
                                let adjusted_axis = if *axis < 0 {
                                    ((*self.shape).len().try_into().unwrap() + *axis)
                                        .try_into()
                                        .unwrap()
                                } else {
                                    (*axis).try_into().unwrap()
                                };
                                axes_arr.append(adjusted_axis);
                            },
                            Option::None => { break; }
                        };
                    };
                    // sort arr
                    bubble_sort(axes_arr.span()).span()
                }
            },
            Option::None => {
                if noop_with_empty_axes {
                    return *self;
                }
                get_all_axes(*self.shape)
            },
        };
        let keepdims = match keepdims {
            Option::Some(keepdims) => keepdims,
            Option::None => true,
        };

        let mut axis_c = 0;
        let mut copy_axes = axes.clone();
        let mut shape = *self.shape;
        let mut data = *self.data;
        loop {
            match copy_axes.pop_front() {
                Option::Some(axis) => {
                    if (shape.len() == 1) {
                        let current_sum = accumulate_sum(data, shape, shape, 0);
                        shape = array![].span();
                        data = array![current_sum].span();
                        break ();
                    }
                    let mut temp_data = array![];
                    let mut temp_shape = reduce_output_shape(shape, *axis - axis_c, false);
                    let data_len = len_from_shape(temp_shape);
                    let mut index: usize = 0;
                    while index != data_len {
                        let indices = unravel_index(index, temp_shape);
                        let current_sum = accumulate_sum(data, shape, indices, *axis - axis_c);

                        temp_data.append(current_sum);

                        index += 1;
                    };

                    shape = temp_shape;
                    data = temp_data.span();
                    axis_c += 1;
                },
                Option::None => { break; }
            };
        };

        let mut axes_copy = axes.clone();
        if keepdims {
            shape = *self.shape;
            loop {
                match axes_copy.pop_front() {
                    Option::Some(axis) => { shape = reduce_output_shape(shape, *axis, true); },
                    Option::None => { break; }
                };
            };
            Tensor { shape, data }
        } else {
            Tensor { shape, data }
        }
    }
}

