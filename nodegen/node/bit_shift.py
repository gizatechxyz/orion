import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait, get_data_statement

class Bit_shift(RunAll):
     
    @staticmethod
    def left_u32():
        x = np.random.randint(0, 60, (3, 3)).astype(np.uint32)
        y = np.random.randint(0, 6, (3, 3)).astype(np.uint32)
        z = x << y

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())
        z = Tensor(Dtype.U32, z.shape, z.flatten())

        name = "bit_shift_left_u32"
        make_test([x, y], z, "TensorTrait::bit_shift(@input_0, @input_1, 'LEFT')", name)
        
    @staticmethod
    def right_u32():
        x = np.random.randint(0, 60, (3, 3)).astype(np.uint32)
        y = np.random.randint(0, 6, (3, 3)).astype(np.uint32)
        z = x >> y

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())
        z = Tensor(Dtype.U32, z.shape, z.flatten())

        name = "bit_shift_right_u32"
        make_test([x, y], z, "TensorTrait::bit_shift(@input_0, @input_1, 'RIGHT')", name)
        