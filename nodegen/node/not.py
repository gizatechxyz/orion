import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test,  Tensor, Dtype


class Not(RunAll):
    @staticmethod
    def not_bool():
        x = np.random.uniform(True, False, (1, 1)).astype(bool)
        y = np.logical_not(x)

        x = Tensor(Dtype.BOOL, x.shape, x.flatten())
        y = Tensor(Dtype.BOOL, y.shape, y.flatten())

        name = "not_bool"
        make_test([x], y, "input_0.not()", name)

    not_bool()
