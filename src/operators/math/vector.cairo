use array::ArrayTrait;
use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;
use onnx_cairo::utils::check_gas;

//=================================================//
//=================== SUM VECTORS =================//
//=================================================//

fn sum_two_vec(vec1: Array::<i32>, vec2: Array::<i32>) -> Array::<i32> {
    assert(vec1.len() == vec2.len(), 'Vectors must have the same size');

    // Initialize variables.
    let mut result = ArrayTrait::new();

    let mut n: usize = 0;
    loop {
        check_gas();

        result.append(*vec1.at(n) + *vec2.at(n));

        n += 1;
        if n == vec1.len() {
            break ();
        };
    };

    return result;
}


//=====================================================//
//=================== FIND MIN IN VECTOR ==============//
//=====================================================//

fn find_min(vec: @Array::<i32>) -> i32 {
    // Initialize variables.
    let mut min_value = IntegerTrait::new(65535_u32, false);

    let mut n: usize = 0;
    loop {
        check_gas();

        let check_min = min_value.min(*vec.at(n));
        if (min_value > check_min) {
            min_value = check_min;
        }

        n += 1;
        if n == vec.len() {
            break ();
        };
    };

    return min_value;
}

//=====================================================//
//=================== FIND MAX IN VECTOR ==============//
//=====================================================//

fn find_max(vec: @Array::<i32>) -> i32 {
    // Initialize variables.
    let mut max_value = IntegerTrait::new(0_u32, false);

    let mut n: usize = 0;
    loop {
        check_gas();

        let check_max = max_value.max(*vec.at(n));
        if (max_value < check_max) {
            max_value = check_max;
        }

        n += 1;
        if n == vec.len() {
            break ();
        };
    };

    return max_value;
}

//=================================================//
//=================== SUM VECTOR ==================//
//=================================================//
fn sum_vec(vec: @Array::<i32>) -> i32 {
    // Initialize variables.
    let mut result = IntegerTrait::new(0_u32, false);

    let mut n: usize = 0;
    loop {
        check_gas();

        result = result + *vec.at(n);

        n += 1;
        if n == vec.len() {
            break ();
        };
    };

    result
}

