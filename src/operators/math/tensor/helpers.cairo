use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;
use onnx_cairo::operators::math::tensor::core::stride;


fn len_from_shape(shape: @Array<usize>, n: usize) -> usize {
    check_gas();
    if (n == shape.len()
        - 1_usize) {
            return *shape.at(n);
        } else {
            return *shape.at(n) * len_from_shape(shape, n + 1_usize);
        }
}

fn check_shape<T>(shape: @Array<usize>, data: @Array<T>) {
    assert(len_from_shape(shape, 0_usize) == data.len(), 'wrong tensor shape');
}

fn check_compatibility(shape_1: @Array<usize>, shape_2: @Array<usize>, index: usize) {
    check_gas();
    assert(shape_1.len() == shape_2.len(), 'tensors shape must match');

    if index == shape_1.len() {
        return ();
    }

    assert(
        *shape_1.at(
            index
        ) == *shape_2.at(index) | *shape_1.at(index) == 1_usize | *shape_2.at(index) == 1_usize,
        'tensors shape must match'
    );

    check_compatibility(shape_1, shape_2, index + 1_usize);
}

fn broadcast_index_mapping(shape: @Array<usize>, indices: @Array<usize>) -> usize {
    let mut result = 0_usize;
    __broadcast_index_mapping(shape, indices, ref result, 0_usize);

    return result;
}

fn __broadcast_index_mapping(
    shape: @Array<usize>, indices: @Array<usize>, ref result: usize, n: usize, 
) {
    check_gas();
    if n == shape.len() {
        return ();
    }

    let stride = stride(shape);
    let index = (*indices.at(n) % *shape.at(n)) * *stride.at(n);
    result += index;
    __broadcast_index_mapping(shape, indices, ref result, n + 1_usize)
}

fn reduce_helper(
    input_shape: @Array<usize>, axis: usize, ref output_shape: Array<usize>, n: usize
) {
    check_gas();
    if n == input_shape.len() {
        return ();
    }

    if n != axis {
        output_shape.append(*input_shape.at(n));
    }
    reduce_helper(input_shape, axis, ref output_shape, n + 1_usize);
}
