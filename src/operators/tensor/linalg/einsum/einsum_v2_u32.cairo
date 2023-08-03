use core::traits::TryInto;
use core::clone::Clone;
use core::option::OptionTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use core::array::SpanTrait;
use core::array::ArrayTrait;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use traits::{Into, Index, IndexView, Default};
use debug::PrintTrait;
use dict::Felt252DictTrait;
use core::felt252;


#[inline(always)]
fn check_gas() {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(_) => {},
        Option::None(_) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        }
    }
}

fn einsum() {
    let mut shape: Array<usize> = ArrayTrait::new();
    let mut data: Array<u32> = ArrayTrait::new();
    let mut result_indices: Array<usize> = ArrayTrait::new();
    // nested_loop();

    if result_indices.len() == 0 {
        let mut data: Array<u32> = array![sum(data)];
    }
}


fn nested_loop(
    n: usize,
    array_n: Span<u32>,
    m: usize,
    array_m: Span<u32>,
    input_tensors_list: Span<Tensor<u32>>,
    ref result_tensor_data: Felt252Dict<u32>,
    input_tensors_indices: Span<Span<usize>>,
    result_indices: Span<usize>,
    n_index: usize,
    m_index: usize,
    mut indices: Array<usize>
) {
    // Run first through the dimensions of output_tensor then on values that are not in output_tensor
    if n_index < n {
        let mut i: usize = 0;
        loop {
            if i == *array_n[n_index] {
                break;
            };

            let mut indices = indices.clone();
            indices.append(i);

            nested_loop(
                n,
                array_n,
                m,
                array_m,
                input_tensors_list,
                ref result_tensor_data,
                input_tensors_indices,
                result_indices,
                n_index + 1,
                m_index,
                indices
            );

            i += 1;
        };
    } else if m_index < m {
        let mut i: usize = 0;
        loop {
            check_gas();
            if i == *array_m[m_index] {
                break;
            };

            let mut indices = indices.clone();
            indices.append(i);

            nested_loop(
                n,
                array_n,
                m,
                array_m,
                input_tensors_list,
                ref result_tensor_data,
                input_tensors_indices,
                result_indices,
                n_index,
                m_index + 1,
                indices
            );

            i += 1;
        };
    } else {
        // Calculate the indices for result_tensor_data based on result_indices
        let mut result_tensor_indices: Array<u32> = ArrayTrait::new();
        let indices = indices.span();
        if result_indices.len() != 0 {
            let mut i = 0;
            loop {
                if i == result_indices.len() {
                    break;
                };
                result_tensor_indices.append(*indices[*result_indices[i]]);
                i += 1;
            }
        }

        // Calculate the indices for each tensor
        let mut tensor_indices_list = ArrayTrait::new();
        let mut i: usize = 0;
        loop {
            if i == input_tensors_indices.len() {
                break;
            };
            let mut tensor_indices = ArrayTrait::new();
            let mut j: usize = 0;
            loop {
                if j == (*input_tensors_indices.at(i)).len() {
                    break;
                };
                tensor_indices.append(*indices[*input_tensors_indices.at(i)[j]]);
                j += 1;
            };
            tensor_indices_list.append(tensor_indices.span());
            i += 1;
        };

        // Perform the multiplication and set the result
        let mut result_value = 1;
        let mut i = 0;
        loop {
            if i == input_tensors_list.len() {
                break;
            };

            result_value *= input_tensors_list[i].at(*tensor_indices_list.span().at(i));
            i += 1;
        };

        // Store Data
        //1. Generate key given indes array (result_tensor_indices)
        let result_len = result_tensor_indices.len();
        let dict_key = encode_array_to_felt252_math(result_tensor_indices);
        //2. Store value(result_value) on dict(dict_key)
        result_tensor_data.insert(dict_key, result_value);
    };
}


fn sum(mut array: Array<u32>) -> u32 {
    let mut x = 0;
    loop {
        if array.len() == 0 {
            break;
        };

        x += array.pop_front().unwrap();
    };

    x
}

// Transform array indices into Dict felt252 key
fn encode_array_to_felt252_math(array: Array<u32>) -> felt252 {
    let mut value: u128 = 0;
    let mut i = array.len();
    let mut power = 1;

    loop {
        if i == 0 {
            break ();
        }

        let add_value: u128 = (*array.at(i - 1)).into() + 1;
        let padded_add_value: u128 = add_value * power;

        value += padded_add_value;

        power = 65536 * power;
        i -= 1;
    };
    return value.into();
}

// Transform a Dict stored indices into a felt252 key
fn dict_to_felt252_key(ref indices: Felt252Dict<u32>, max: u32) -> felt252 {
    let mut j = 0;
    let mut key_value: u128 = 0;
    let mut power = 1;
    loop {
        if j > max - 1 {
            break ();
        }
        let indice = indices.get(j.into()) + 1;
        let diff = max - j - 1;

        key_value += indice.into() * power;
        power = 65536 * power;

        j += 1;
    };

    return key_value.into();
}

fn dict_to_tensor_array_converter(
    indices_array: Array<usize>,
    ref indices_counter: Felt252Dict<u32>,
    ref original_dict: Felt252Dict<u32>,
    ref output_array: Array<u32>,
    depth: usize,
) {
    // If we've reached the maximum depth, print indices
    if depth == indices_array.len() {
        let key_value = dict_to_felt252_key(ref indices_counter, depth);
        let value = original_dict.get(key_value);
        output_array.append(value);
    } else {

        let mut i = 0;
        loop {
            let indice_max = *indices_array[depth];
            if i > indice_max {
                break;
            }
            // Recurse to the next depth
            dict_to_tensor_array_converter(
                indices_array.clone(),
                ref indices_counter,
                ref original_dict,
                ref output_array,
                depth + 1
            );

            // Increment the current index
            let current_indice_i = indices_counter.get(depth.into());
            indices_counter.insert(depth.into(), current_indice_i + 1);
            i += 1;
        };

        // Reset the current index before returning to the previous depth
        let depth_felt252: felt252 = depth.into();
        indices_counter.insert(depth_felt252, 0);
    }
}


#[test]
#[available_gas(20000000)]
fn dict_to_tensor_test() {
    let extra = Option::<ExtraParams>::None(());
    let result_tensor = TensorTrait::<u32>::new(
        array![3, 4].span(), array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].span(), extra
    );
    let mut indices: Felt252Dict<u32> = Default::default();
    let mut original_dict: Felt252Dict<u32> = Default::default();
    original_dict.insert(4295032833, 100);
    original_dict.insert(8590000129, 200);

    let mut output_array = ArrayTrait::new();

    dict_to_tensor_array_converter(
        array![4, 2, 2], ref indices, ref original_dict, ref output_array, 0
    );

    assert(*output_array[0] == 100, 'Output should be 100');
    assert(*output_array[1] == 200, 'Output should be 200');

}


#[test]
#[available_gas(20000000)]
fn einsum_matmul() {
    let extra = Option::<ExtraParams>::None(());
    let a_tensor = TensorTrait::<u32>::new(
        array![3, 4].span(),
        array![173, 99, 145, 176, 41, 10, 123, 168, 27, 103, 165, 214].span(),
        extra
    );

    let extra = Option::<ExtraParams>::None(());
    let b_tensor = TensorTrait::<u32>::new(
        array![4, 2].span(), array![149, 197, 80, 225, 142, 138, 85, 187].span(), extra
    );

    // let mut c_data = ArrayTrait::new();
    let mut indices = ArrayTrait::new();
    let mut c_data: Felt252Dict<u32> = Default::default();

    nested_loop(
        2,
        array![3, 2].span(),
        1,
        array![4].span(),
        array![a_tensor, b_tensor].span(),
        ref c_data,
        array![array![0, 2].span(), array![2, 1].span()].span(),
        array![0, 1].span(),
        0,
        0,
        indices
    );

    let mut tensor_array: Array<usize> = ArrayTrait::new();
    let mut output_index_array = array![3, 4];
    let mut indices: Felt252Dict<u32> = Default::default();

    dict_to_tensor_array_converter(
        output_index_array, ref indices, ref c_data, ref tensor_array, 0
    );

    let mut x = 0;
    'Printing Tensor Array'.print();
    loop {
        if x == tensor_array.len() {
            break;
        }

        (*tensor_array.at(x)).print();

        x += 1;
    }
}

