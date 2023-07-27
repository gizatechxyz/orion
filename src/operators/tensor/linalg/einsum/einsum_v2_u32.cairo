use core::clone::Clone;
use core::option::OptionTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use core::array::SpanTrait;
use core::array::ArrayTrait;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use traits::{Into, Index, IndexView, Default};
use debug::PrintTrait;

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
    ref result_tensor_data: Array<u32>,
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
        result_tensor_data.append(result_value);

        result_value.print();
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

    let mut c_data = ArrayTrait::new();
    let mut indices = ArrayTrait::new();

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
// let mut x = 0;
// loop {
//     if x == c_data.len() {
//         break;
//     }

//     (*c_data.at(x)).print();

//     x += 1;
// }
}
