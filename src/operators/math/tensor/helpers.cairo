use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;
use onnx_cairo::operators::math::tensor::core::stride;

fn len_from_shape(shape: Span<usize>) -> usize {
    let mut result: usize = 1;

    let mut i: usize = 0;
    loop {
        check_gas();

        if i == shape.len() {
            break ();
        }

        result *= *shape.at(i);
        i += 1;
    };

    return result;
}

fn check_shape<T>(shape: Span<usize>, data: @Array<T>) {
    assert(len_from_shape(shape) == data.len(), 'wrong tensor shape');
}

fn check_compatibility(shape_1: Span<usize>, shape_2: Span<usize>) {
    assert(shape_1.len() == shape_2.len(), 'tensors shape must match');

    let mut n: usize = 0;
    loop {
        check_gas();

        assert(
            *shape_1.at(
                n
            ) == *shape_2.at(n) | *shape_1.at(n) == 1_usize | *shape_2.at(n) == 1_usize,
            'tensors shape must match'
        );

        n += 1;
        if n == shape_1.len() {
            break ();
        };
    };
}

fn broadcast_index_mapping(shape: Span<usize>, indices: Span<usize>) -> usize {
    let mut result = 0_usize;

    let mut n: usize = 0;
    loop {
        check_gas();

        let stride = stride(shape);
        let index = (*indices.at(n) % *shape.at(n)) * *stride.at(n);
        result += index;

        n += 1;
        if n == shape.len() {
            break ();
        };
    };

    return result;
}

fn reduce_helper(input_shape: Span<usize>, axis: usize) -> Span<usize> {
    let mut reduced = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        if n != axis {
            reduced.append(*input_shape.at(n));
        }

        n += 1;
        if n == input_shape.len() {
            break ();
        };
    };

    return reduced.span();
}
