use core::clone::Clone;
use core::option::OptionTrait;
use traits::{Into, Index, IndexView, Default};
use array::{ArrayTrait, SpanTrait};
use dict::Felt252DictTrait;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use debug::PrintTrait;


fn einsum(
    inputs_eq: @Array<Array<felt252>>, output_eq: @Array<felt252>, inputs: @Array<Tensor<u32>>
) {
    // Check that the number of inputs matches the number of inputs in the equation
    assert(inputs.len() == inputs_eq.len(), 'Missmatch inputs with equation');

    // Builds a map from tensor indices (as represented in the equation string) to their sizes
    // This is crucial for verifying that the tensors provided as input to the function match 
    // the sizes that the equation string expects.
    let mut indices_to_size: Felt252Dict<u32> = Default::default();
    let mut i: usize = 0;
    loop {
        if i == inputs.len() {
            break;
        };

        let mut j: usize = 0;
        loop {
            if j == inputs_eq[i].len() {
                break;
            };

            let c = inputs_eq.at(i)[j];
            if indices_to_size.get(*c) == 0 {
                indices_to_size.insert(*c, *(*inputs.at(i)).shape[j])
            } else {
                if indices_to_size.get(*c) != *(*inputs.at(i)).shape[j] {
                    panic(array!['DimMismatch']);
                }
            }
            j += 1;
        };

        i += 1;
    };

    // Iterates over the indices in the output equation. 
    // For each index, if the index is not already in the indices_to_size map, 
    // it inserts the index with a size of 1.    
    let mut i: usize = 0;
    loop {
        if i == output_eq.len() {
            break;
        };

        if indices_to_size.get(*output_eq[i]) == 0 {
            indices_to_size.insert(*output_eq[i], 1);
        }

        i += 1;
    };

    // Compute the output tensor shape
    let mut output_shape: Array<u32> = ArrayTrait::new();
    let mut i: usize = 0;
    loop {
        if i == output_eq.len() {
            break;
        };

        let c = output_eq.at(i);
        if indices_to_size.get(*c) != 0 {
            output_shape.append(indices_to_size.get(*c));
        }

        i += 1;
    };

    if output_shape.len() == 0 {
        output_shape.append(1);
    };
    let output_shape: Span<u32> = output_shape.span();

    // Initialize an empty dictionary for seen indices
    let mut seen: Felt252Dict<u8> = Default::default();
    // Initialize an empty array for common indices
    let mut common_indices_to_inputs: Array<felt252> = ArrayTrait::new();
    let mut i: usize = 0;
    loop {
        if i == inputs_eq.len() {
            break;
        };

        let mut j: usize = 0;
        loop {
            if j == inputs_eq.at(i).len() {
                break;
            };

            let c = inputs_eq.at(i)[j];
            if seen.get(*c) == 0 {
                // Here we insert the index 'c' into the dictionary with value 'true'
                seen.insert(*c, 1);
            } else {
                common_indices_to_inputs.append(*c);
            }

            j += 1;
        };

        i += 1;
    };

    // Initialize the cartesian_coord as an empty array of arrays.
    let mut cartesian_coord: Array<Array<u32>> = ArrayTrait::new();
    // Iterate over each dimension in the output_shape
    let mut i = 0;
    loop {
        if i >= output_shape.len() {
            break;
        };

        // For each dimension, generate an array representing the range from 0 to the dimension size
        let mut range: Array<u32> = ArrayTrait::new();
        let mut j: usize = 0;
        loop {
            if j >= *output_shape.at(i) {
                break;
            }

            range.append(j);

            j += 1;
        };

        // Append this range to cartesian_coord.
        cartesian_coord.append(range);

        i += 1;
    };
    // Call cartesian_product function to generate the actual cartesian product.
    cartesian_coord = cartesian_product(cartesian_coord);
// TODO: Compute the cartesian product of all indices
// TODO: Create Tensor
}

// output
// for (0, i)
// for ((0, j))

// input
// for( (0. k))
// for (0, l)

// out[i][j] = 

fn cartesian_product_recursive(
    mut remaining: Array::<Array<u32>>, current: Array<u32>, ref result: Array::<Array<u32>>
) {
    if remaining.len() == 0 {
        // No more lists to process, push the current product to the result
        result.append(current)
    } else {
        // Still have lists to process
        let first: Array<u32> = remaining.pop_front().unwrap();
        let mut i = 0;
        loop {
            if i >= first.len() {
                break;
            };
            // Add the current item to the current product
            let mut next_current = current.clone();
            next_current.append(*first[i]);
            // Recurse on the rest of the lists
            cartesian_product_recursive(remaining.clone(), next_current, ref result);
            i += 1;
        }
    }
}

fn cartesian_product(lists: Array<Array<u32>>) -> Array<Array<u32>> {
    let mut result = ArrayTrait::new();
    cartesian_product_recursive(lists, ArrayTrait::new(), ref result);

    result
}

