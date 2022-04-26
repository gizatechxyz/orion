
from onnx_cairo.small_math import (
    Fix64x61,
    Double,
    Double_to_Fix,
    show_Double,
    Small_Math_mul,
    Small_Math_add
)

from starkware.cairo.common.alloc import alloc

################ TENSOR ELEMENT TO ELEMENT OPERATIONS ###############

################### FELT OPERATIONS #################################
func arrays_add_felt (array_1: felt*, array_2: felt*, size: felt) -> (res: felt*):
    alloc_locals
    let (local res: felt*) = alloc()
    arrays_add_felt__inner (array_1, array_2, res, size)
    return(res = res)
end

func arrays_add_felt__inner (array_1: felt*, array_2: felt*, res: felt*, size: felt):
    if size == 0:
        return()
    end
    assert [res] = [array_1] + [array_2]
    arrays_add_felt__inner (array_1 + 1, array_2 + 1, res + 1, size - 1)
    return()
end

func arrays_mul_felt (array_1: felt*, array_2: felt*, size: felt) -> (res: felt*):
    alloc_locals
    let (local res: felt*) = alloc()
    arrays_mul_felt__inner (array_1, array_2, res, size)
    return(res = res)
end

func arrays_mul_felt__inner (array_1: felt*, array_2: felt*, res: felt*, size: felt):
    if size == 0:
        return()
    end
    assert [res] = [array_1] * [array_2]
    arrays_mul_felt__inner (array_1 + 1, array_2 + 1, res + 1, size - 1)
    return()
end

################### FIX OPERATIONS #################################
func arrays_mul_fix {range_check_ptr}(array_1: Fix64x61*, array_2: Fix64x61*, size: felt) -> (res: Fix64x61*):
    alloc_locals
    let (local res: Fix64x61*) = alloc()
    arrays_mul_felt__inner (array_1, array_2, res, size)
    return(res = res)
end

func arrays_mul_fix__inner {range_check_ptr}(array_1: Fix64x61*, array_2: Fix64x61*, res: Fix64x61*, size: felt):
    if size == 0:
        return()
    end
    let array_1_elem = [array_1]
    let array_2_elem = [array_2]
    let (mul_res) = Small_Math_mul (array_1_elem, array_2_elem)
    assert [res] = mul_res
    arrays_mul_felt__inner (array_1 + Fix64x61.SIZE, array_2 + Fix64x61.SIZE, res + 1, size - 1)
    return()
end

func arrays_add_fix {range_check_ptr}(array_1: Fix64x61*, array_2: Fix64x61*, size: felt) -> (res: Fix64x61*):
    alloc_locals
    let (local res: Fix64x61*) = alloc()
    arrays_add_fix__inner (array_1, array_2, res, size)
    return(res = res)
end

func arrays_add_fix__inner {range_check_ptr}(array_1: Fix64x61*, array_2: Fix64x61*, res: Fix64x61*, size: felt):
    if size == 0:
        return()
    end
    let array_1_elem = [array_1]
    let array_2_elem = [array_2]
    let (mul_res) = Small_Math_add (array_1_elem, array_2_elem)
    assert [res] = mul_res
    arrays_add_fix__inner (array_1 + Fix64x61.SIZE, array_2 + Fix64x61.SIZE, res + 1, size - 1)
    return()
end