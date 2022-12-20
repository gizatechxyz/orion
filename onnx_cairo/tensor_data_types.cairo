from starkware.cairo.common.alloc import alloc

from contracts.onnx_cairo.small_math import (
    Fix64x61,
    Double,
    Double_to_Fix,
)

struct Tensor:
    member dims_size: felt
    member dims: felt*
    member vars: felt*
    member elements_size: felt
    member elements: felt*
end

struct TensorFix:
    member dims_size: felt
    member dims: felt*
    member vars: felt*
    member elements_size: felt
    member elements: Fix64x61*
end


################ FIX TENSOR INIT #######################
func init_tensor_from_double {range_check_ptr}(dim_size: felt, dims: felt*, elements_size: felt, elements_double: Double*) -> (res: TensorFix):
    alloc_locals
    let vars: felt* = alloc()
    let elements_fix: Fix64x61* = alloc()
    array_from_double_to_fix (elements_double, elements_fix, elements_size, 0)
    let res: TensorFix = TensorFix (dim_size, dims, vars, elements_size, elements_fix)
    init_tensor_from_double__load_vars (res, 0)
    return (res = res)
end

func array_from_double_to_fix {range_check_ptr}(array_double: Double*, array_fix: Fix64x61*, size: felt, index: felt):
    if index == size:
        return()
    end
    let current_double: Double = [array_double + index * Double.SIZE]
    let (new_fix: Fix64x61) = Double_to_Fix (current_double)
    assert [array_fix + index * Fix64x61.SIZE] = new_fix
    array_from_double_to_fix (array_double, array_fix, size, index + 1)
    return()
end

func init_tensor_from_double__load_vars (tensor: TensorFix, index: felt):
    if index == tensor.dims_size:
        return()
    end
    let tensor_dims : felt* = tensor.dims
    let tensor_vars : felt* = tensor.vars
    let (var) = init_tensor__multiply_n_first (tensor_dims, index)
    assert [tensor_vars + index] = var
    init_tensor_from_double__load_vars (tensor, index + 1)
    return()
end

################ FELT TENSOR INIT #######################
func init_tensor_fix (dim_size: felt, dims: felt*, elements_size: felt, elements: Fix64x61*) -> (res: TensorFix):
    alloc_locals
    let vars: felt* = alloc()
    let res: TensorFix = TensorFix (dim_size, dims, vars, elements_size, elements)
    init_tensor_fix__load_vars (res, 0)
    return (res = res)
end

func init_tensor_fix__load_vars (tensor: TensorFix, index: felt):
    if index == tensor.dims_size:
        return()
    end
    let tensor_dims : felt* = tensor.dims
    let tensor_vars : felt* = tensor.vars
    let (var) = init_tensor__multiply_n_first (tensor_dims, index)
    assert [tensor_vars + index] = var
    init_tensor_fix__load_vars (tensor, index + 1)
    return()
end

################ FELT TENSOR INIT #######################
func init_tensor (dim_size: felt, dims: felt*, elements_size: felt, elements: felt*) -> (res: Tensor):
    alloc_locals
    let vars: felt* = alloc()
    let res: Tensor = Tensor (dim_size, dims, vars, elements_size, elements)
    init_tensor__load_vars (res, 0)
    return (res = res)
end

func init_tensor__load_vars (tensor: Tensor, index: felt):
    if index == tensor.dims_size:
        return()
    end
    let tensor_dims : felt* = tensor.dims
    let tensor_vars : felt* = tensor.vars
    let (var) = init_tensor__multiply_n_first (tensor_dims, index)
    assert [tensor_vars + index] = var
    init_tensor__load_vars (tensor, index + 1)
    return()
end

func init_tensor__multiply_n_first (elements: felt*, elements_size: felt) -> (res: felt):
    if elements_size == 0:
        return(res = 1)
    end
    let (partial_sum) = init_tensor__multiply_n_first(elements + 1, elements_size - 1)
    tempvar res_temp = [elements]
    tempvar res = partial_sum * res_temp
    return (res)
end