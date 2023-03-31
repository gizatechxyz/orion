use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;


fn len_from_shape(shape: @Array<usize>, index: usize) -> usize {
    check_gas();
    if (index == shape.len()
        - 1_usize) {
            return *shape.at(index);
        } else {
            return *shape.at(index) * len_from_shape(shape, index + 1_usize);
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