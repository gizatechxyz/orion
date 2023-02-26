from starkware.cairo.common.alloc import alloc

from onnx_cairo.small_math import Fix64x61, Double, Double_to_Fix, Small_Math_mul, Small_Math_add

from onnx_cairo.tensor_data_types import Tensor, TensorFix, init_tensor, init_tensor_fix

// Error handling

// error = 0
//   non-specific error

// error = 1
//   dimensions not right

//

func error_coder(error: felt) -> (res: Tensor) {
    alloc_locals;
    let (dims: felt*) = alloc();

    assert ([dims]) = 0;

    let (elements: felt*) = alloc();
    assert ([elements]) = error;

    let (res: Tensor) = init_tensor(0, dims, 1, elements);
    return (res=res);
}

func error_coder_fix(error: felt) -> (res: TensorFix) {
    alloc_locals;
    let (dims: felt*) = alloc();

    assert ([dims]) = 0;

    let (elements: Fix64x61*) = alloc();
    assert ([elements]) = Fix64x61(val=error);

    let (res: TensorFix) = init_tensor_fix(0, dims, 1, elements);
    return (res=res);
}
