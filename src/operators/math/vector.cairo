use array::ArrayTrait;
use onnx_cairo::operators::math::int33;
use onnx_cairo::operators::math::int33::i33;

//=================================================//
//=================== SUM VECTORS =================//
//=================================================//

fn sum_two_vec(vec1: Array::<i33>, vec2: Array::<i33>) -> Array::<i33> {
    // Initialize variables.
    let mut _vec1 = vec1;
    let mut _vec2 = vec2;
    let mut result = ArrayTrait::new();

    __sum_two_vec(ref _vec1, ref _vec2, ref result, 0_usize);

    return result;
}

fn __sum_two_vec(
    ref vec1: Array::<i33>, ref vec2: Array::<i33>, ref result: Array::<i33>, n: usize, 
) {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    assert(vec1.len() == vec2.len(), 'Vectors must have the same size');

    // --- End of the recursion ---
    if n == vec1.len() {
        return ();
    }

    // --- Sum and assign the result to the current index ---
    result.append(*vec1.at(n) + *vec2.at(n));

    // --- The process is repeated for the remaining elemets in the array --- 
    __sum_two_vec(ref vec1, ref vec2, ref result, n + 1_usize);
}


//=================================================//
//=================== FIND IN VECTOR ==============//
//=================================================//

fn find_min_max(vec: @Array::<i33>) -> (i33, i33) {
    // Initialize variables.
    let mut min_value = i33 { inner: 65535_u32, sign: false };
    let mut max_value = i33 { inner: 65535_u32, sign: true };

    __find_min_max(vec, ref min_value, ref max_value, 0_usize, );

    return (min_value, max_value);
}

fn __find_min_max(vec: @Array::<i33>, ref min_value: i33, ref max_value: i33, n: usize, ) {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    // --- End of the recursion ---
    if n == vec.len() {
        return ();
    }

    // --- Check the minimum value and update min_value if necessary --- 
    let check_min = int33::min(min_value, *vec.at(n));
    if (min_value != check_min) {
        min_value = check_min;
    }

    // --- Check the maximum value and update max_value if necessary --- 
    let check_max = int33::max(max_value, *vec.at(n));
    if (max_value != check_max) {
        max_value = check_max;
    }

    // --- The process is repeated for the remaining elemets in the array --- 
    __find_min_max(vec, ref min_value, ref max_value, n + 1_usize);
}

//=====================================================//
//=================== FIND MIN IN VECTOR ==============//
//=====================================================//

fn find_min(vec: @Array::<i33>) -> i33 {
    // Initialize variables.
    let mut min_value = i33 { inner: 65535_u32, sign: false };

    __find_min(vec, ref min_value, 0_usize);

    return min_value;
}

fn __find_min(vec: @Array::<i33>, ref min_value: i33, n: usize) {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    // --- End of the recursion ---
    if n == vec.len() {
        return ();
    }

    // --- Check the minimum value and update min_value if necessary --- 
    let check_min = int33::min(min_value, *vec.at(n));
    if (min_value != check_min) {
        min_value = check_min;
    }

    // --- The process is repeated for the remaining elemets in the array --- 
    __find_min(vec, ref min_value, n + 1_usize);
}


//=================================================//
//=================== SUM VECTOR ==================//
//=================================================//
fn sum_vec(vec: @Array::<i33>) -> i33 {
    // Initialize variables.
    let mut result = i33 { inner: 0_u32, sign: false };

    __sum_vec(vec, ref result, 0_usize);

    result
}

fn __sum_vec(vec: @Array::<i33>, ref result: i33, n: usize) {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    // --- End of the recursion ---
    if n == vec.len() {
        return ();
    }

    result = result + *vec.at(n);
    __sum_vec(vec, ref result, n + 1_usize);
}
