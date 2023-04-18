use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;

use onnx_cairo::operators::math::signed_integer;
use onnx_cairo::operators::math::signed_integer::i32;
use onnx_cairo::operators::math::vector::find_min;
use onnx_cairo::operators::math::vector::sum_vec;
use onnx_cairo::utils::check_gas;

// The implementation is using pseudo-softmax for now: 
// x = (x - min(x)) / sum(x)
// pseudo-softmax: subtract by min value and divide by sum -> less sparse but similar properties as softmax
fn softmax(z: @Matrix) -> Matrix {
    let mut arr = ArrayTrait::<i32>::new();
    let min = find_min(z.data);
    let sum = sum_vec(z.data);

    let mut index: usize = 0;
    loop {
        check_gas();
        arr.append((*z.data.at(index) - min) / sum);
        index += 1;
        if index == z.len() {
            break ();
        };
    };

    MatrixTrait::new(*z.rows, *z.cols, arr)
}