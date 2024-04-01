use alexandria_data_structures::array_ext::SpanTraitExt;

use orion::numbers::NumberTrait;
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

/// Cf: TensorTrait::gather_nd docstring
fn gather_nd<T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,>(
    self: @Tensor<T>, indices: Tensor<usize>, batch_dims: Option<usize>
) -> Tensor<T> {
    let batch_dims = match batch_dims {
        Option::Some(val) => val,
        Option::None => 0
    };

    let data_rank = (*self.shape).len();
    let indices_rank = (indices.shape).len();
    assert((data_rank >= 1) & (indices_rank >= 1), 'rank must > 1');

    let mut data_shape = *self.shape;
    let mut indices_shape = indices.shape;
    let mut data_shape_clone = data_shape.clone();
    let mut indices_shape_clone = indices_shape.clone();

    let indices_shape_last = indices_shape_clone.pop_back().unwrap();
    assert(
        (*indices_shape_last >= 1) & (*indices_shape_last <= data_rank - batch_dims),
        'check indices'
    );

    let mut batch_dims_shape = array![];
    let mut output_shape = array![];
    let mut index_data = array![];
    let mut output_data = array![];

    let mut batch_dims_size = batch_dims;
    let mut total_data_len = 1;
    let mut multiple_data_len = array![];

    let mut ind = 0;
    while ind != batch_dims {
        match indices_shape_clone.pop_front() {
            Option::Some(val) => {
                batch_dims_size *= *val;
                batch_dims_shape.append(*val);
                ind += 1;
            },
            Option::None => { break; }
        };
    };

    loop {
        match indices_shape_clone.pop_front() {
            Option::Some(val) => { batch_dims_shape.append(*val); },
            Option::None => { break; }
        };
    };

    if (*indices_shape_last == data_rank - batch_dims) {
        output_shape = batch_dims_shape;
    } else {
        let mut ind = 0;
        output_shape = batch_dims_shape;
        loop {
            match data_shape_clone.pop_front() {
                Option::Some(val) => {
                    if (ind >= (batch_dims + *indices_shape_last)) {
                        output_shape.append(*val);
                    }

                    ind += 1;
                },
                Option::None => { break; }
            };
        };
    }

    let mut ind = 0;
    let mut multiple = 1;
    let mut incrementer = 1;
    let mut data_shape_clone = data_shape.clone();
    loop {
        match data_shape_clone.pop_front() {
            Option::Some(val) => {
                if (ind >= batch_dims) {
                    multiple *= *val;
                    multiple_data_len.append(multiple);
                }

                if (ind >= batch_dims + *indices_shape_last) {
                    incrementer *= *val;
                }

                ind += 1;
            },
            Option::None => { break; }
        };
    };

    let mut ind = 0;
    let mut indices_shape_clone = indices_shape.clone();
    let mut breaker = 1;
    loop {
        match indices_shape_clone.pop_front() {
            Option::Some(val) => {
                if (ind >= batch_dims) {
                    breaker *= *val;
                }

                ind += 1;
            },
            Option::None => { break; }
        };
    };

    total_data_len = *multiple_data_len.at(multiple_data_len.len() - 1);
    let mut data_indices = indices.data;
    let mut ind = 0;
    let mut result = 0;
    loop {
        match data_indices.pop_front() {
            Option::Some(val) => {
                let index = ind % *indices_shape_last;
                let incr = total_data_len * (ind / breaker);
                result += (*val * total_data_len / *multiple_data_len.at(index));
                ind += 1;

                if (index == *indices_shape_last - 1) {
                    let mut data_ind: usize = result;
                    while data_ind != result
                        + incrementer {
                            index_data.append(data_ind + incr);
                            data_ind += 1;
                        };

                    result = 0;
                };
            },
            Option::None => { break; }
        };
    };

    loop {
        match index_data.pop_front() {
            Option::Some(val) => { output_data.append(*self.data[val]); },
            Option::None => { break; }
        };
    };

    let mut output_tensor = TensorTrait::<T>::new(output_shape.span(), output_data.span());

    output_tensor
}
