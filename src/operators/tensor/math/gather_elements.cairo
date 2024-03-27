use core::option::OptionTrait;
use core::traits::TryInto;
use alexandria_data_structures::array_ext::SpanTraitExt;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{{TensorTrait, Tensor}, core::{unravel_index, stride}};

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

    let data_strides = stride(*self.shape);

    let mut output_data = array![];
    let mut i: usize = 0;
    while i < indices
        .data
        .len() {
            let indice = *indices.data.at(i);
            let adjusted_indice: u32 = if indice < 0 {
                ((*(*self.shape).at(axis)).try_into().unwrap() + indice).try_into().unwrap()
            } else {
                indice.try_into().unwrap()
            };

            assert(adjusted_indice < (*(*self.shape).at(axis)), 'Index out of bounds');

            let multidim_index = unravel_index(i, indices.shape);
            let mut flat_index_for_data = 0;

            let mut j: usize = 0;
            while j < multidim_index
                .len() {
                    let dim_index = *multidim_index.at(j);
                    if j == axis {
                        flat_index_for_data += adjusted_indice * (*data_strides.at(j));
                    } else {
                        flat_index_for_data += (dim_index * *data_strides.at(j))
                    }
                    j += 1;
                };

            assert(
                flat_index_for_data < (*self.data).len().try_into().unwrap(),
                'Flat index out of bounds'
            );

            output_data.append(*(*self.data).at(flat_index_for_data));
            i += 1;
        };

    TensorTrait::<T>::new(indices.shape, output_data.span())
}
