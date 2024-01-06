import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, Dtype, Tensor, Trait


class Sequence_empty(RunAll):

    @staticmethod
    def sequence_empty_u32():
        def default():
            shape=(0,)
            x = np.zeros(shape, dtype=np.uint32)
            t = Tensor(Dtype.U32, shape, x.flatten())
            make_test(
                inputs=[],
                output=[t],
                func_sig="SequenceTrait::sequence_empty()",
                name="sequence_empty_u32",
                trait=Trait.SEQUENCE
            )

        default()

    @staticmethod
    def sequence_empty_i32():
        def default():
            shape=(0,)
            x = np.zeros(shape, dtype=np.int32)
            t = Tensor(Dtype.I32, shape, x.flatten())
            make_test(
                inputs=[],
                output=[t],
                func_sig="SequenceTrait::sequence_empty()",
                name="sequence_empty_i32",
                trait=Trait.SEQUENCE
            )

        default()

    @staticmethod
    def sequence_empty_i8():
        def default():
            shape=(0,)
            x = np.zeros(shape, dtype=np.int8)
            t = Tensor(Dtype.I8, shape, x.flatten())
            make_test(
                inputs=[],
                output=[t],
                func_sig="SequenceTrait::sequence_empty()",
                name="sequence_empty_i8",
                trait=Trait.SEQUENCE
            )

        default()

    @staticmethod
    def sequence_empty_fp8x23():
        def default():
            shape=(0,)
            x = np.zeros(shape, dtype=np.float64)
            t = Tensor(Dtype.FP8x23, shape, x.flatten())
            make_test(
                inputs=[],
                output=[t],
                func_sig="SequenceTrait::sequence_empty()",
                name="sequence_empty_fp8x23",
                trait=Trait.SEQUENCE
            )

        default()

    @staticmethod
    def sequence_empty_fp16x16():
        def default():
            shape=(0,)
            x = np.zeros(shape, dtype=np.float64)
            t = Tensor(Dtype.FP16x16, shape, x.flatten())
            make_test(
                inputs=[],
                output=[t],
                func_sig="SequenceTrait::sequence_empty()",
                name="sequence_empty_fp16x16",
                trait=Trait.SEQUENCE
            )

        default()
