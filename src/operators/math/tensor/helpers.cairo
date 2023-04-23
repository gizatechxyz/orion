use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::utils::check_gas;
use onnx_cairo::operators::math::tensor::core::stride;

// Calculates the number of elements in a tensor given its shape
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

// Verifies if the shape and the data array of a tensor are compatible
fn check_shape<T>(shape: Span<usize>, data: @Array<T>) {
    assert(len_from_shape(shape) == data.len(), 'wrong tensor shape');
}

// Checks if two tensor shapes are compatible for broadcasting
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

// Computes the index in the broadcasted tensor corresponding to the given indices and shape
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

// Generates the output shape after reducing a tensor along a specified axis
fn reduce_output_shape(input_shape: Span<usize>, axis: usize) -> Span<usize> {
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

// Helper function that computes the output shape of a tensor after applying the axes permutation
fn permutation_output_shape(input_shape: Span<usize>, axes: @Array<usize>) -> Span<usize> {
    let mut output_shape = ArrayTrait::new();
    let mut axis: usize = 0;

    loop {
        check_gas();
        if axis == axes.len() {
            break ();
        }

        output_shape.append(*input_shape.at(*axes.at(axis)));
        axis += 1;
    };

    return output_shape.span();
}

// Combines output indices with the current index of the specified axis
fn combine_indices(output_indices: Span<usize>, axis_index: usize, axis: usize) -> Span<usize> {
    let mut result = ArrayTrait::new();
    let output_indices_len = output_indices.len();
    let mut n: usize = 0;

    loop {
        check_gas();

        if n > output_indices_len {
            break ();
        }

        if n == axis {
            result.append(axis_index);
        } else if n > axis {
            result.append(*output_indices.at(n - 1_usize));
        } else {
            result.append(*output_indices.at(n));
        }

        n += 1;
    };

    return result.span();
}


// Helper function that finds the index of a target axis in the given axes array
fn find_axis(axes: @Array<usize>, target_axis: usize) -> usize {
    let mut axis: usize = 0;
    loop {
        check_gas();
        if axis == axes.len() {
            break ();
        }

        if *axes.at(axis) == target_axis {
            break ();
        }
        axis += 1;
    };
    return axis;
}
