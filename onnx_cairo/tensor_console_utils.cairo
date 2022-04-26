from starkware.cairo.common.alloc import alloc

from onnx_cairo.tensor_data_types import (
    Tensor,
    TensorFix,
    init_tensor
)

func show_array (array: felt*, size: felt, tabs: felt):
    show_array_inner (array, size + 1, 1, tabs)
    return()
end

func show_array_inner (array: felt*, size: felt, index: felt, tabs: felt):
    if index == size:
        return()
    end
    tempvar element = [array + index - 1]
    %{
        text = str(ids.index) + '/' + str(ids.size - 1) + ': ' + str(ids.element)
        print ('\t'*ids.tabs, text)
    %}
    show_array_inner(array, size, index + 1, tabs)
    return()
end

func show_tensor(tensor: Tensor):
    if tensor.dims_size == 0:
        tempvar tensor_elements= tensor.elements
        tempvar scalar = [tensor_elements]
        %{  
            print (' Scalar:', ids.scalar)
        %}
        return()
    end
    tempvar dims_size = tensor.dims_size
    %{
        print (' dims_size:', ids.dims_size)
        print (' dims:')
    %}
    show_array (tensor.dims, tensor.dims_size, 1)
    %{
        print (' vars:')
    %}
    show_array (tensor.vars, tensor.dims_size, 1)
    tempvar tensor_elements_size = tensor.elements_size
    %{
        print (' elements_size:', ids. tensor_elements_size)
        print (' elements:')
    %}
    show_array (tensor.elements, tensor.elements_size, 1)
    return()
end

func show_error (error_tensor: Tensor):
    let tensor_elements = error_tensor.elements
    tempvar error = [tensor_elements]
    if error == 0:
        %{
            print (' Error code 0: Non-specific error')
        %}
        return()
    end
    if error == 1:
        %{
            print (' Error code 1: Dimensions do not match')
        %}
        return()
    end
    return()
end

func show_error_fix (error_tensor: TensorFix):
    let tensor_elements = error_tensor.elements
    let tensor_elements_fix = [tensor_elements]
    tempvar error = tensor_elements_fix.val
    if error == 0:
        %{
            print (' Error code 0: Non-specific error')
        %}
        return()
    end
    if error == 1:
        %{
            print (' Error code 1: Dimensions do not match')
        %}
        return()
    end
    return()
end
    