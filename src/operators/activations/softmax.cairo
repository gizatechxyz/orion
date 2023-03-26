use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;

use onnx_cairo::operators::math::int33;
use onnx_cairo::operators::math::int33::i33;

use onnx_cairo::operators::math::vector::find_min;
use onnx_cairo::operators::math::vector::sum_vec;

// The implementation is using pseudo-softmax for now: 
// x = (x - min(x)) / sum(x)
// pseudo-softmax: subtract by min value and divide by sum -> less sparse but similar properties as softmax
fn softmax(z: @Matrix) -> Matrix {
    let mut arr = ArrayTrait::<i33>::new();

    let min = find_min(z.data);

    let mut sum = sum_vec(z.data);

    _softmax(z.data, ref arr, min, sum, 0_usize);

    MatrixTrait::new(*z.rows, *z.cols, arr)
}

fn _softmax(z: @Array::<i33>, ref arr: Array::<i33>, min: i33, sum: i33, index: usize) {
    if index == z.len() {
        return ();
    }

    let mut result = *z.at(index) - min;
    result = result / sum;
    arr.append(result);

    _softmax(z, ref arr, min, sum, index + 1_usize)
}
