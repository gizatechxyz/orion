import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, Tensor, Dtype

class Not(RunAll):
    @staticmethod
    def not_bool():
        x = np.random.uniform(True, False, (1, 1)).astype(bool)
        y = ~(x)

        x = Tensor(Dtype.Bool, x.shape, x.flatten())
        y = Tensor(Dtype.Bool, y.shape, y.flatten())


        name = "not_bool"
        make_node([x], [y], name)
        make_test([x], y, "input_0", name)