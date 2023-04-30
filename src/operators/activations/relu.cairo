use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;
use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;
use onnx_cairo::utils::check_gas;

fn relu(z: @Matrix) -> Matrix {
    let mut arr = ArrayTrait::<i32>::new();

    let mut index: usize = 0;
    loop {
        check_gas();
        let val_0 = IntegerTrait::new(0, false);

        if *z.data.at(index) > val_0 {
            arr.append(*z.data.at(index));
        } else {
            arr.append(val_0);
        }

        index += 1;
        if index == z.data.len() {
            break ();
        };
    };

    MatrixTrait::new(*z.rows, *z.cols, arr)
}
