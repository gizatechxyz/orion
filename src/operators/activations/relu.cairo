use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;

use onnx_cairo::operators::math::int33;
use onnx_cairo::operators::math::int33::i33;


fn relu(z: @Matrix) -> Matrix {
    let mut arr = ArrayTrait::<i33>::new();

    relu_inner(ref arr, z.data, 0_usize, z.data.len());
    MatrixTrait::new(*z.rows, *z.cols, arr)
}

fn relu_inner(ref arr: Array::<i33>, input: @Array::<i33>, index: usize, len: usize) {
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

    let val_0 = (i33 { inner: 0_u32, sign: true });

    // if x > 0 -> x
    if *input.at(
        index
    ) > val_0 {
        arr.append(*input.at(index));
    } // if x < 0 -> 0
    else {
        arr.append(val_0);
    }
    relu_inner(ref arr, input, index + 1_usize, len);
}
