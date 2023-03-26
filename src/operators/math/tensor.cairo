use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::int33;
use onnx_cairo::operators::math::int33::i33;
use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;

impl Arrayi33Drop of Drop::<Array::<i33>>;

// #[derive(Drop)]
struct Tensor {
    rows: usize,
    cols: usize,
    depth: usize,
    data: Array::<Matrix>,
}

trait TensorTrait {
    fn new(rows: usize, cols: usize, depth: usize, data: Array::<Matrix>) -> Tensor;
}

impl TensorImpl of TensorTrait {
    #[inline(always)]
    fn new(rows: usize, cols: usize, depth: usize, data: Array::<Matrix>) -> Tensor {
        assert(data.len() == rows * cols * depth, 'Tensor not match dimensions');
        tensor_new(rows, cols, depth, data)
    }
}

fn tensor_new(rows: usize, cols: usize, depth: usize, data: Array::<Matrix>) -> Tensor {
    Tensor { rows: rows, cols: cols, depth: depth, data: data,  }
}
