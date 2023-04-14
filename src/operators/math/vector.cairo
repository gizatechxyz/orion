use array::ArrayTrait;
use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;

//=================================================//
//=================== SUM VECTORS =================//
//=================================================//

fn sum_two_vec(vec1: Array::<i32>, vec2: Array::<i32>) -> Array::<i32> {
    // Initialize variables.
    let mut _vec1 = vec1;
    let mut _vec2 = vec2;
    let mut result = ArrayTrait::new();

    __sum_two_vec(ref _vec1, ref _vec2, ref result, 0_usize);

    return result;
}

fn __sum_two_vec(
    ref vec1: Array::<i32>, ref vec2: Array::<i32>, ref result: Array::<i32>, n: usize, 
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

//=====================================================//
//=================== FIND MIN IN VECTOR ==============//
//=====================================================//

fn find_min(vec: @Array::<i32>) -> i32 {
    // Initialize variables.
    let mut min_value = IntegerTrait::new(65535_u32, false);

    __find_min(vec, ref min_value, 0_usize);

    return min_value;
}

fn __find_min(vec: @Array::<i32>, ref min_value: i32, n: usize) {
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
    let check_min = min_value.min(*vec.at(n));
    if (min_value > check_min) {
        min_value = check_min;
    }

    // --- The process is repeated for the remaining elemets in the array --- 
    __find_min(vec, ref min_value, n + 1_usize);
}

//=====================================================//
//=================== FIND MAX IN VECTOR ==============//
//=====================================================//

fn find_max(vec: @Array::<i32>) -> i32 {
    // Initialize variables.
    let mut max_value = IntegerTrait::new(0_u32, false);

    __find_max(vec, ref max_value, 0_usize);

    return max_value;
}

fn __find_max(vec: @Array::<i32>, ref max_value: i32, n: usize) {
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
    let check_max = max_value.max(*vec.at(n));
    if (max_value < check_max) {
        max_value = check_max;
    }

    // --- The process is repeated for the remaining elemets in the array --- 
    __find_max(vec, ref max_value, n + 1_usize);
}


//=================================================//
//=================== SUM VECTOR ==================//
//=================================================//
fn sum_vec(vec: @Array::<i32>) -> i32 {
    // Initialize variables.
    let mut result = IntegerTrait::new(0_u32, false);

    __sum_vec(vec, ref result, 0_usize);

    result
}

fn __sum_vec(vec: @Array::<i32>, ref result: i32, n: usize) {
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
