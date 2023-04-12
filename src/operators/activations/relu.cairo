use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;

use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;


fn relu(z: @Matrix) -> Matrix {
    let mut arr = ArrayTrait::<i32>::new();

    relu_mag(ref arr, z.data, 0_usize, z.data.len());
    MatrixTrait::new(*z.rows, *z.cols, arr)
}

fn relu_mag(ref arr: Array::<i32>, input: @Array::<i32>, index: usize, len: usize) {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    if index == len {
        return ();
    }

    let val_0 = IntegerTrait::new(0_u32, false);

    // if x > 0 -> x
    if *input.at(
        index
    ) > val_0 {
        arr.append(*input.at(index));
    } // if x < 0 -> 0
    else {
        arr.append(val_0);
    }
    relu_mag(ref arr, input, index + 1_usize, len);
}
