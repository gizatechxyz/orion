
from starkware.cairo.common.alloc import alloc

from onnx_cairo.tensor_data_types import (
    Tensor,
    init_tensor
)

func generate_random_array (size: felt, range_bottom: felt, range_top: felt) -> (res: felt*):
    alloc_locals
    let res: felt* = alloc()
    generate_random_array__inner (res, size, range_bottom, range_top)
    return (res = res)
end

func generate_random_array__inner (res: felt*, size: felt, range_bottom: felt, range_top: felt):
    alloc_locals
    if size == 0:
        return()
    end
    local rnd_felt: felt
    %{
        import random
        ids.rnd_felt = random.randint(ids.range_bottom, ids.range_top)
    %}
    assert [res] = rnd_felt 
    generate_random_array__inner (res + 1, size - 1, range_bottom, range_top)
    return()
end

func generate_shift_value_array (size: felt, shift: felt) -> (res: felt*):
    alloc_locals
    let res: felt* = alloc()
    generate_shift_value_array__inner (res, size, 0, shift)
    return (res = res)
end

func generate_shift_value_array__inner (res: felt*, size: felt, index: felt, shift: felt):
    alloc_locals
    if index == size:
        return()
    end
    assert [res] = shift + index 
    generate_shift_value_array__inner (res + 1, size, index + 1, shift)
    return()
end

func load_tensor_a () -> (res: Tensor):
    alloc_locals
    let (dims: felt*) = alloc()
    
    assert ([dims]) = 2
    assert ([dims + 1]) = 3
    assert ([dims + 2]) = 4

    let (elements: felt*) = alloc()
    assert ([elements]) = 1
    assert ([elements + 1]) = 2
    assert ([elements + 2]) = 3
    assert ([elements + 3]) = 4
    assert ([elements + 4]) = 5
    assert ([elements + 5]) = 6
    assert ([elements + 6]) = 7
    assert ([elements + 7]) = 8
    assert ([elements + 8]) = 11
    assert ([elements + 9]) = 12
    assert ([elements + 10]) = 13
    assert ([elements + 11]) = 14
    assert ([elements + 12]) = 15
    assert ([elements + 13]) = 16
    assert ([elements + 14]) = 17
    assert ([elements + 15]) = 18
    assert ([elements + 16]) = 21
    assert ([elements + 17]) = 22
    assert ([elements + 18]) = 23
    assert ([elements + 19]) = 24
    assert ([elements + 20]) = 25
    assert ([elements + 21]) = 26
    assert ([elements + 22]) = 27
    assert ([elements + 23]) = 28

    let (res: Tensor) = init_tensor (3, dims, 24, elements)
    return (res = res)
end

func load_vector_dim5_1 () -> (res: Tensor):
    alloc_locals
    let (dims: felt*) = alloc()
    
    assert ([dims]) = 5


    let (elements: felt*) = alloc()
    assert ([elements]) = 1
    assert ([elements + 1]) = 2
    assert ([elements + 2]) = 3
    assert ([elements + 3]) = 4
    assert ([elements + 4]) = 5

    let (res: Tensor) = init_tensor (1, dims, 5, elements)
    return (res = res)
end

func load_vector_dim5_2 () -> (res: Tensor):
    alloc_locals
    let (dims: felt*) = alloc()
    
    assert ([dims]) = 5


    let (elements: felt*) = alloc()
    assert ([elements]) = 6
    assert ([elements + 1]) = 7
    assert ([elements + 2]) = 8
    assert ([elements + 3]) = 9
    assert ([elements + 4]) = 10

    let (res: Tensor) = init_tensor (1, dims, 5, elements)
    return (res = res)
end
