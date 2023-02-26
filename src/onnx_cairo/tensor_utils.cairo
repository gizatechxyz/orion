// %builtins range_check

from starkware.cairo.common.alloc import alloc

from onnx_cairo.tensor_data_types import (
    Tensor,
    TensorFix,
    init_tensor,
    init_tensor_fix,
    init_tensor_from_double,
)

from onnx_cairo.tensor_operations import (
    arrays_add_felt,
    arrays_add_fix,
    arrays_mul_felt,
    arrays_mul_fix,
)

from onnx_cairo.tensor_error_handler import error_coder, error_coder_fix

from onnx_cairo.small_math import Fix64x61, Double, Double_to_Fix

// INDEX OPERATIONS

// Cutting
// Rearranging
// Pairing / Selecting (filtering?)
// Dot operations
// Elem to elem operations
// Checking operations (dimensiones compatibility)

// ##################### FELT WRAPPERS FOR MATH OPERATIONS ##########################
func elem_x_elem_add(tensor_1: Tensor, tensor_2: Tensor) -> (res: Tensor) {
    let (check_equal_dims) = check_elem_x_elem_req(tensor_1, tensor_2);
    if (check_equal_dims == 0) {
        let (res_tensor) = error_coder(1);
        return (res=res_tensor);
    }
    let (res_elements: felt*) = arrays_add_felt(
        tensor_1.elements, tensor_2.elements, tensor_1.elements_size
    );
    let (res: Tensor) = init_tensor(
        tensor_1.dims_size, tensor_1.dims, tensor_1.elements_size, res_elements
    );
    return (res=res);
}

func elem_x_elem_mul(tensor_1: Tensor, tensor_2: Tensor) -> (res: Tensor) {
    let (check_equal_dims) = check_elem_x_elem_req(tensor_1, tensor_2);
    if (check_equal_dims == 0) {
        let (res_tensor) = error_coder(1);
        return (res=res_tensor);
    }
    let (res_elements: felt*) = arrays_mul_felt(
        tensor_1.elements, tensor_2.elements, tensor_1.elements_size
    );
    let (res: Tensor) = init_tensor(
        tensor_1.dims_size, tensor_1.dims, tensor_1.elements_size, res_elements
    );
    return (res=res);
}

// ##################### FIX WRAPPERS FOR MATH OPERATIONS ##########################
func elem_x_elem_fix_add{range_check_ptr}(tensor_1: TensorFix, tensor_2: TensorFix) -> (
    res: TensorFix
) {
    alloc_locals;
    let (check_equal_dims) = check_elem_x_elem_fix_req(tensor_1, tensor_2);
    if (check_equal_dims == 0) {
        let (res_tensor) = error_coder_fix(1);
        return (res=res_tensor);
    }
    let (res_elements: Fix64x61*) = arrays_add_fix(
        tensor_1.elements, tensor_2.elements, tensor_1.elements_size
    );
    let (res: TensorFix) = init_tensor_fix(
        tensor_1.dims_size, tensor_1.dims, tensor_1.elements_size, res_elements
    );
    return (res=res);
}

func elem_x_elem_fix_mul{range_check_ptr}(tensor_1: TensorFix, tensor_2: TensorFix) -> (
    res: TensorFix
) {
    alloc_locals;
    let (check_equal_dims) = check_elem_x_elem_fix_req(tensor_1, tensor_2);
    if (check_equal_dims == 0) {
        let (res_tensor) = error_coder_fix(1);
        return (res=res_tensor);
    }
    let (res_elements: Fix64x61*) = arrays_mul_fix(
        tensor_1.elements, tensor_2.elements, tensor_1.elements_size
    );
    let (res: TensorFix) = init_tensor_fix(
        tensor_1.dims_size, tensor_1.dims, tensor_1.elements_size, res_elements
    );
    return (res=res);
}

// ###################### CHECK REQUIREMENTS UTIL #####################################
func check_elem_x_elem_fix_req(tensor_1: TensorFix, tensor_2: TensorFix) -> (res: felt) {
    let (check_condition_dims) = equal_arrays(tensor_1.dims, tensor_2.dims, tensor_1.dims_size);
    tempvar check_condition_sizes = (tensor_1.dims_size - tensor_2.dims_size + 1) + (
        tensor_1.elements_size - tensor_2.elements_size + 1
    );
    tempvar res = check_condition_sizes * check_condition_dims;
    return (res=res);
}

func check_elem_x_elem_req(tensor_1: Tensor, tensor_2: Tensor) -> (res: felt) {
    let (check_condition_dims) = equal_arrays(tensor_1.dims, tensor_2.dims, tensor_1.dims_size);
    tempvar check_condition_sizes = (tensor_1.dims_size - tensor_2.dims_size + 1) + (
        tensor_1.elements_size - tensor_2.elements_size + 1
    );
    tempvar res = check_condition_sizes * check_condition_dims;
    return (res=res);
}

// ################## COMPARING TENSORS UTIL ###############################
func check_equal_tensors(tensor_1: Tensor, tensor_2: Tensor) -> (res: felt) {
    alloc_locals;
    let (local check_condition_dims) = equal_arrays(
        tensor_1.dims, tensor_2.dims, tensor_1.dims_size
    );
    let (local check_condition_elements) = equal_arrays(
        tensor_1.elements, tensor_2.elements, tensor_1.elements_size
    );
    tempvar check_condition_sizes = (tensor_1.dims_size - tensor_2.dims_size + 1) + (
        tensor_1.elements_size - tensor_2.elements_size + 1
    );
    tempvar res = check_condition_sizes * check_condition_dims * check_condition_elements;
    return (res=res);
}

func equal_arrays(array_1: felt*, array_2: felt*, size: felt) -> (res: felt) {
    if (size == 0) {
        return (1,);
    }
    if ([array_1] != [array_2]) {
        return (0,);
    }
    let (res) = equal_arrays(array_1 + 1, array_2 + 1, size - 1);
    return (res,);
}

// ########## WRAPPER FOR FIX - AXIS_SUM ########################################3
func axis_sum_fix(tensor_fix: TensorFix, dims_permutation: felt*) -> (res: TensorFix) {
    alloc_locals;
    let elements_felt: felt* = cast(tensor_fix.elements, felt*);
    let tensor_felt: Tensor = Tensor(
        tensor_fix.dims_size,
        tensor_fix.dims,
        tensor_fix.vars,
        tensor_fix.elements_size,
        elements_felt,
    );
    let (axis_sum_res) = axis_sum(tensor_felt, dims_permutation);
    let elements_res: Fix64x61* = cast(axis_sum_res.elements, Fix64x61*);
    let tensor_res_fix: TensorFix = TensorFix(
        axis_sum_res.dims_size,
        axis_sum_res.dims,
        axis_sum_res.vars,
        axis_sum_res.elements_size,
        elements_res,
    );
    return (res=tensor_res_fix);
}

// AXIS_SUM FUNCTION
func axis_sum(tensor: Tensor, dims_permutation: felt*) -> (res: Tensor) {
    alloc_locals;
    let (looped_indexes) = loop(tensor, dims_permutation);
    let looped_indexes_elements: felt* = looped_indexes.elements;
    let (tensor_elements_permuted) = permutate_array(
        tensor.elements, looped_indexes_elements, tensor.elements_size
    );
    tempvar sliced_index = [dims_permutation + tensor.dims_size - 1];
    let tensor_dims = tensor.dims;
    tempvar sum_vector_size = [tensor_dims + sliced_index];
    let (res_elements: felt*) = alloc();
    axis_sum__inner(
        tensor_elements_permuted,
        res_elements,
        looped_indexes_elements,
        tensor.elements_size,
        sum_vector_size,
        0,
    );
    let (new_dims) = permutate_array(tensor.dims, dims_permutation, tensor.dims_size);
    let new_elements_size = tensor.elements_size / sum_vector_size;
    let (res: Tensor) = init_tensor(
        tensor.dims_size - 1, new_dims, new_elements_size, res_elements
    );
    return (res=res);
}

func axis_sum__inner(
    tensor_elements_permuted: felt*,
    res: felt*,
    looped_indexes: felt*,
    looped_indexes_size: felt,
    sum_size: felt,
    index: felt,
) {
    if (index == looped_indexes_size) {
        return ();
    }
    let (vector_sum) = sum_n_elements(tensor_elements_permuted, looped_indexes, index, sum_size);
    tempvar res_index = index / sum_size;
    assert [res + res_index] = vector_sum;
    axis_sum__inner(
        tensor_elements_permuted,
        res,
        looped_indexes,
        looped_indexes_size,
        sum_size,
        index + sum_size,
    );
    return ();
}

func sum_n_elements(
    tensor_elements_permuted: felt*, looped_indexes: felt*, initial_index: felt, n: felt
) -> (res: felt) {
    alloc_locals;
    if (n == 0) {
        return (res=0);
    }
    local current_val = [tensor_elements_permuted + initial_index + n - 1];
    let (next_val) = sum_n_elements(tensor_elements_permuted, looped_indexes, initial_index, n - 1);
    return (res=current_val + next_val);
}

func loop(tensor: Tensor, dims_permutation: felt*) -> (res: Tensor) {
    alloc_locals;
    let res_elements: felt* = alloc();
    assert [res_elements] = 0;
    let (res_elements: felt*) = loop__inner(
        tensor, res_elements, 1, dims_permutation, tensor.dims_size, 0
    );
    let (res_dims: felt*) = permutate_array(tensor.dims, dims_permutation, tensor.dims_size);
    let (res: Tensor) = init_tensor(tensor.dims_size, res_dims, tensor.elements_size, res_elements);
    return (res=res);
}

// current_expansion initial = [0]
func loop__inner(
    tensor: Tensor,
    current_expansion: felt*,
    current_expansion_size: felt,
    dims_permutation: felt*,
    permutation_size: felt,
    permutation_index: felt,
) -> (res_elements: felt*) {
    alloc_locals;
    if (permutation_index == permutation_size) {
        return (current_expansion,);
    }
    local dims_index = dims_permutation[permutation_index];
    local dims_array: felt* = tensor.dims;
    local vars_array: felt* = tensor.vars;
    local current_dim = [dims_array + dims_index];
    local current_var = [vars_array + dims_index];
    let (expanded_elements: felt*) = alloc();
    expand_vector(
        expanded_elements,
        tensor,
        current_expansion,
        current_expansion_size,
        current_dim,
        current_var,
        0,
        0,
    );
    tempvar expanded_elements_size = current_expansion_size * current_dim;
    let (res_elements) = loop__inner(
        tensor,
        expanded_elements,
        expanded_elements_size,
        dims_permutation,
        permutation_size,
        permutation_index + 1,
    );
    return (res_elements=res_elements);
}

// GET VECTOR FROM TENSOR AND EXPAND VECTORS ------------------------------------------------------------
func expand_vector(
    res: felt*,
    tensor: Tensor,
    initial_elements: felt*,
    initial_elements_size: felt,
    vector_size: felt,
    step: felt,
    res_size: felt,
    index_initial_elements: felt,
) {
    if (index_initial_elements == initial_elements_size) {
        return ();
    }
    tempvar initial_element = [initial_elements + index_initial_elements];
    let (vector_to_add) = get_index_vector_from_tensor(tensor, initial_element, vector_size, step);
    append_array(res, vector_to_add, res_size, vector_size);
    expand_vector(
        res,
        tensor,
        initial_elements,
        initial_elements_size,
        vector_size,
        step,
        res_size + vector_size,
        index_initial_elements + 1,
    );
    return ();
}

// TODO: just pass dimensions, not the whole tensor
// GETS VECTORS OF INDEXES
func get_index_vector_from_tensor(
    tensor: Tensor, initial_element: felt, vector_size: felt, step: felt
) -> (res: felt*) {
    alloc_locals;
    let res: felt* = alloc();
    get_index_vector_from_tensor__inner(
        tensor.elements, initial_element, res, vector_size, step, 0
    );
    return (res=res);
}

func get_index_vector_from_tensor__inner(
    tensor_elements: felt*, initial_element: felt, res: felt*, res_size: felt, step: felt, index
) {
    if (index == res_size) {
        return ();
    }
    tempvar index_in_tensor = initial_element + index * step;
    assert [res + index] = index_in_tensor;
    get_index_vector_from_tensor__inner(
        tensor_elements, initial_element, res, res_size, step, index + 1
    );
    return ();
}

// GETS VECTORS OF ACTUAL VALUES IN THE TENSOR
func get_vector_from_tensor(
    tensor: Tensor, initial_element: felt, vector_size: felt, step: felt
) -> (res: felt*) {
    alloc_locals;
    let res: felt* = alloc();
    get_vector_from_tensor__inner(tensor.elements, initial_element, res, vector_size, step, 0);
    return (res=res);
}

func get_vector_from_tensor__inner(
    tensor_elements: felt*, initial_element: felt, res: felt*, res_size: felt, step: felt, index
) {
    if (index == res_size) {
        return ();
    }
    tempvar index_in_tensor = initial_element + index * step;
    assert [res + index] = [tensor_elements + index_in_tensor];
    get_vector_from_tensor__inner(tensor_elements, initial_element, res, res_size, step, index + 1);
    return ();
}

// APPEND ARRAY ---------------------------------------------------------------
func append_array(array_1: felt*, array_2: felt*, array_1_size: felt, array_2_size: felt) {
    if (array_2_size == 0) {
        return ();
    }
    append_array(array_1, array_2, array_1_size, array_2_size - 1);
    assert [array_1 + array_1_size + array_2_size - 1] = [array_2 + array_2_size - 1];
    return ();
}

// PERMUTATE ARRAY ---------------------------------------------------------------
func permutate_array(array: felt*, permutation: felt*, size: felt) -> (res: felt*) {
    alloc_locals;
    let res: felt* = alloc();
    permutate_array__inner(array, permutation, res, size);
    return (res=res);
}

func permutate_array__inner(array: felt*, permutation: felt*, res: felt*, size: felt) {
    if (size == 0) {
        return ();
    }
    tempvar permutation_val = [permutation];
    assert [res] = [array + permutation_val];
    permutate_array__inner(array, permutation + 1, res + 1, size - 1);
    return ();
}

// REMOVE ELEMENT -------------------------------------------------------------------
func remove_element(array: felt*, size: felt, index_removed: felt) -> (res: felt*) {
    alloc_locals;
    let res: felt* = alloc();
    let () = remove_element__inner(array, size, 0, res, 0, index_removed);
    return (res=res);
}

func remove_element__inner(
    array: felt*, size: felt, index_ini: felt, res: felt*, index_fin: felt, index_removed: felt
) {
    if (index_ini == size) {
        return ();
    }
    if (index_ini == index_removed) {
        remove_element__inner(array, size, index_ini + 1, res, index_fin, index_removed);
    }
    if (index_ini != index_removed) {
        assert [res + index_fin] = [array + index_ini];
        remove_element__inner(array, size, index_ini + 1, res, index_fin + 1, index_removed);
    }
    return ();
}

func main{range_check_ptr}() {
    alloc_locals;

    // # SUM ELEMENT X ELEMENT
    // let tensor_1: Tensor = load_tensor_a()
    // show_tensor (tensor_1)
    // let tensor_2: Tensor = load_tensor_a()
    // show_tensor (tensor_2)
    // let (res_tensor: Tensor) = elem_x_elem_mul(tensor_1, tensor_2)
    // show_tensor(res_tensor)

    // # GENERATE RANDOM ARRAY
    // let (vector_1) = generate_random_array(7, 1, 36)
    // show_array (vector_1, 7, 0)
    // let (removed_vector) = remove_element (vector_1, 7, 3)
    // show_array (removed_vector, 6, 1)

    // # PERMUTATE AN ARRAY
    // let (vector_1) = generate_random_array(7, 1, 36)
    // show_array (vector_1, 7, 0)
    // let permutation_1 : felt* = alloc()
    // assert  [permutation_1] = 2
    // assert  [permutation_1 + 1] = 1
    // assert  [permutation_1 + 2] = 6
    // assert  [permutation_1 + 3] = 4
    // assert  [permutation_1 + 4] = 0
    // assert  [permutation_1 + 5] = 3
    // assert  [permutation_1 + 6] = 5
    // show_array (permutation_1, 7, 1)
    // let (permutated_array: felt*) = permutate_array(vector_1, permutation_1, 7)
    // show_array (permutated_array, 7 , 2)

    // # APPEND ARRAY
    // let (vector_1) = generate_random_array(5, 0, 4)
    // show_array (vector_1, 5, 0)
    // let (vector_2) = generate_random_array(5, 5, 9)
    // show_array (vector_2, 5, 1)
    // append_array (vector_1, vector_2, 5, 5)
    // show_array (vector_1, 10, 2)

    // # GET VECTOR FROM TENSOR
    // let (vector_1) = generate_random_array(20, 1, 20)
    // show_array (vector_1, 20, 0)
    // let dims: felt* = alloc()
    // assert [dims] = 4
    // assert [dims + 1] = 5
    // let (initial_vector: felt*) = alloc()
    // assert [initial_vector] = 1
    // assert [initial_vector + 1] = 2
    // assert [initial_vector + 2] = 3
    // let (random_tensor_1: Tensor) = init_tensor(2, dims, 20, vector_1)
    // let (res: felt*) = alloc()
    // expand_vector (res, random_tensor_1, initial_vector, 3, 5, 4, 0, 0)
    // show_array (res, 15, 1)

    // # TESTING AXIS_SUM FUNC
    // let (vector_1) = generate_shift_value_array(30, 100)
    // let dims: felt* = alloc()
    // assert [dims] = 5
    // assert [dims + 1] = 3
    // assert [dims + 2] = 2
    // let (random_tensor_1: Tensor) = init_tensor(3, dims, 30, vector_1)
    // %{
    //     print(' Tensor:')
    // %}
    // show_tensor (random_tensor_1)
    // let permutation_1 : felt* = alloc()
    // assert  [permutation_1] = 1
    // assert  [permutation_1 + 1] = 2
    // assert  [permutation_1 + 2] = 0
    // %{
    //     print(' Permutation:')
    // %}
    // show_array (permutation_1, 3, 3)
    // let (res_tensor: Tensor) = axis_sum (random_tensor_1, permutation_1)
    // %{
    //     print(' showing res_tensor:')
    // %}
    // show_tensor (res_tensor)

    // TESTING INIT_TENSOR CON DOUBLES
    let dims: felt* = alloc();
    assert [dims] = 2;
    assert [dims + 1] = 3;
    let elements_double: Double* = alloc();
    assert [elements_double] = Double(8965, 3);
    assert [elements_double + Double.SIZE] = Double(1653, 3);
    assert [elements_double + 2 * Double.SIZE] = Double(7653, 3);
    assert [elements_double + 3 * Double.SIZE] = Double(4869, 3);
    assert [elements_double + 4 * Double.SIZE] = Double(2237, 3);
    assert [elements_double + 5 * Double.SIZE] = Double(6547, 3);
    let tensor_1: TensorFix = init_tensor_from_double(2, dims, 6, elements_double);

    let elements_fix: Fix64x61* = tensor_1.elements;
    let element_fix_1: Fix64x61 = [elements_fix];
    let element_fix_2: Fix64x61 = [elements_fix + 1];
    let element_fix_3: Fix64x61 = [elements_fix + 2];
    let element_fix_4: Fix64x61 = [elements_fix + 3];
    let element_fix_5: Fix64x61 = [elements_fix + 4];
    let element_fix_6: Fix64x61 = [elements_fix + 5];
    let permutation_1: felt* = alloc();
    assert [permutation_1] = 1;
    assert [permutation_1 + 1] = 0;
    let (axis_sum_fix_res) = axis_sum_fix(tensor_1, permutation_1);
    let elements_res_fix: Fix64x61* = axis_sum_fix_res.elements;
    let element_res_fix_1: Fix64x61 = [elements_res_fix];
    let element_res_fix_2: Fix64x61 = [elements_res_fix + 1];
    let element_res_fix_3: Fix64x61 = [elements_res_fix + 2];
    let (tensor_res_add) = elem_x_elem_fix_add(tensor_1, tensor_1);
    let elements_add_fix: Fix64x61* = tensor_res_add.elements;
    let element_add_fix_1: Fix64x61 = [elements_add_fix];
    let element_add_fix_2: Fix64x61 = [elements_add_fix + 1];
    let element_add_fix_3: Fix64x61 = [elements_add_fix + 2];
    let element_add_fix_4: Fix64x61 = [elements_add_fix + 3];
    let element_add_fix_5: Fix64x61 = [elements_add_fix + 4];
    let element_add_fix_6: Fix64x61 = [elements_add_fix + 5];

    let (tensor_res_mul) = elem_x_elem_fix_mul(tensor_1, tensor_1);
    let elements_mul_fix: Fix64x61* = tensor_res_mul.elements;
    let element_mul_fix_1: Fix64x61 = [elements_mul_fix];
    let element_mul_fix_2: Fix64x61 = [elements_mul_fix + 1];
    let element_mul_fix_3: Fix64x61 = [elements_mul_fix + 2];
    let element_mul_fix_4: Fix64x61 = [elements_mul_fix + 3];
    let element_mul_fix_5: Fix64x61 = [elements_mul_fix + 4];
    let element_mul_fix_6: Fix64x61 = [elements_mul_fix + 5];
    return ();
}
