use alexandria_data_structures::array_ext::SpanTraitExt;
use array::ArrayTrait;
use array::SpanTrait;

use core::traits::Into;
use debug::PrintTrait;
use core::traits::TryInto;
use core::serde::Serde;
use core::traits::Destruct;
use option::OptionTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

/// Cf: TensorTrait::gather docstring
fn gather_elements<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    // impl TAddEq: AddEq<T>,
    // impl TMulEq: MulEq<T>,
    // impl TPartialOrd: PartialOrd<T>,
    // impl TPartialEq: PartialEq<T>,
>(
    self: @Tensor<T>, indices: Tensor<usize>, axis: Option<usize>
) -> Tensor<T> {
    let axis = match axis {
        Option::Some(val) => val,
        Option::None(_) => 0
    };
    assert(axis < (*self.shape).len(), 'axis out of dimensions');

    let data_rank = (*self.shape).len();
    let indices_rank = (indices.shape).len();
    assert((data_rank == indices_rank ) & (indices_rank >= 1), 'must be same rank');


    let axis_shape = *(*self.shape).at(axis);
    let ind_max = indices.data.max().unwrap();
    assert(ind_max < axis_shape, 'this index out of bounds');


    // let ind_max = indices.data.max().unwrap();
    // assert(ind_max < axis_shape, 'this index out of bounds');
    let mut indices_shape = indices.shape;
    let data_shape = *self.shape;


    if (indices_rank == 2) {
        if (axis == 0) {
            assert(indices_shape.at(axis+1) == data_shape.at(axis+1), 'out of index')
        }
        if (axis == 1) {
            assert(indices_shape.at(axis-1) == data_shape.at(axis-1), 'out of index')
        }
    }

    // if (indices_rank == 3) {
    //     if (axis == 0) {
    //         assert((indices_shape.at(axis+1) == data_shape.at(axis+1)) & (indices_shape.at(axis+2) == data_shape.at(axis+2)), 'out of index')
    //     }
    //     if (axis == 1) {
    //         assert((*indices_shape.at(axis-1) < *data_shape.at(axis-1)) & (indices_shape.at(axis+1) == data_shape.at(axis+1)), 'out of index')
    //     }
    //     if (axis == 2) {
    //         assert((*indices_shape.at(axis-2) <= *data_shape.at(axis-2)) & (indices_shape.at(axis-1) == data_shape.at(axis-1)), 'out of index')
    //     }
    // }

    let mut output_data = ArrayTrait::new();

    let mut outer_loop = indices_shape.at(axis);
    let mut inner_loop = 1;
    let mut multiplier = 1;
    let mut ind = 0;
    loop {
        match indices_shape.pop_front() {
            Option::Some(val) => {
                inner_loop *= *val;
                if (ind >= axis) {
                    multiplier *= *val;
                }
               
                ind += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let looper = multiplier / *outer_loop;

    if inner_loop != 1 {
        inner_loop /= *outer_loop;
    }

    

    let mut data_indices = indices.data;
    let mut i: usize = 0;
    loop {
        match data_indices.pop_front() {
            Option::Some(val) => {
                if (axis == 0){
                    let value  = *val * inner_loop.into() + (i % inner_loop);
                    output_data.append(*self.data[value]);
                }
                if ((axis == indices_rank-1) & (axis != 0)) {
                    let value = *val + *outer_loop * (i / *outer_loop);
                    output_data.append(*self.data[value]);

                }
                 if ((axis != indices_rank-1) & (axis != 0)) {
                    let value = *val * (looper ) + (i % looper) + (multiplier  * (i / multiplier));
                    // 'start'.print();
                    // i.print();
                    // (*val).print();
                    // (looper).print();
                    // (i % multiplier).print();
                    // (multiplier).print();
                    // (i / multiplier).print();
                    // value.print();

                    output_data.append(*self.data[value]);
                }
                i += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let mut output_tensor = TensorTrait::<T>::new(indices.shape, output_data.span());
    return output_tensor;
}


// Tests--------------------------------------------------------------------------------------------------------------


use orion::numbers::fixed_point::implementations::fp8x23::helpers::assert_precise;
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::I32TensorPartialEq;
use orion::utils::assert_eq;


fn indices1() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(5);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(2);
    data.append(2);
    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn indices() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(0);
    data.append(1);

    TensorTrait::new(shape.span(), data.span())
}

fn indices2() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);


    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(2);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(1);
    data.append(0);
    data.append(2);
    data.append(1);
    data.append(1);
    data.append(2);
    data.append(1);
    data.append(0);
    data.append(2);
    data.append(1);

    TensorTrait::new(shape.span(), data.span())
}

fn indices22() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);


    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(1);

    TensorTrait::new(shape.span(), data.span())
}

fn data1() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(5);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn data() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
   

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn data2() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    data.append(10);
    data.append(11);
    data.append(12);
    data.append(13);
    data.append(14);
    data.append(15);
    data.append(16);
    data.append(17);

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn data3() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    data.append(10);
    data.append(11);
    data.append(12);
    data.append(13);
    data.append(14);
    data.append(15);
    data.append(16);
    data.append(17);
    data.append(18);
    data.append(19);
    data.append(20);
    data.append(21);
    data.append(22);
    data.append(23);
    data.append(24);
    data.append(25);
    data.append(26);

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn u32_tensor_3x2x3x2_index() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(3);
    sizes.append(2);


    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(0);

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn u32_tensor_3x2x3x2_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(3);
    sizes.append(2);


    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    data.append(10);
    data.append(11);
    data.append(12);
    data.append(13);
    data.append(14);
    data.append(15);
    data.append(16);
    data.append(17);
    data.append(18);
    data.append(19);
    data.append(20);
    data.append(21);
    data.append(22);
    data.append(23);
    data.append(24);
    data.append(25);
    data.append(26);
    data.append(27);
    data.append(28);
    data.append(29);
    data.append(30);
    data.append(31);
    data.append(32);
    data.append(33);
    data.append(34);
    data.append(35);

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}



#[test]
#[available_gas(20000000000)]
fn test_gather_elements_default() {
    let data = u32_tensor_3x2x3x2_helper();
    let indices = u32_tensor_3x2x3x2_index();


    let y = data.gather_elements(indices: indices,  axis:Option::Some(2));
    let mut output = y.data;

    loop {
        match output.pop_front() {
            Option::Some(val) => {
               (*val).print();
            },
            Option::None(_) => { break; }
        };
    };

}

#[test]
#[available_gas(20000000000)]
fn test_gather_elements_axis_1() {
    let data = data2();
    let indices = indices22();

    let y = data.gather_elements(indices: indices,  axis:Option::Some(1));
    let mut output = y.data;

    assert(*output[0] == 0, 'not correct');
    assert(*output[1] == 3, 'not correct');
    assert(*output[2] == 2, 'not correct');
    assert(*output[3] == 3, 'not correct');
    assert(*output[4] == 2, 'not correct');
    assert(*output[5] == 1, 'not correct');
    assert(*output[6] == 8, 'not correct');
    assert(*output[7] == 9, 'not correct');
    assert(*output[8] == 8, 'not correct');
    assert(*output[9] == 7, 'not correct');
    assert(*output[10] == 8, 'not correct');
    assert(*output[11] == 9, 'not correct');


    let y = data.gather_elements(indices: indices,  axis:Option::Some(1));
    let mut output = y.data;

   

}

// #[test]
// #[available_gas(2000000000)]
// fn test_gather_elements_3_lastaxis() {
//     let data = data();
//     let indices = indices();


//     let y = data.gather_elements(indices: indices,  axis:Option::Some(1));
//     let mut output = y.data;

//     assert(*output[0] == 1, 'not correct');
//     assert(*output[1] == 2, 'not correct');
//     assert(*output[2] == 3, 'not correct');
//     assert(*output[3] == 5, 'not correct');
//     assert(*output[4] == 4, 'not correct');
//     assert(*output[5] == 5, 'not correct');

// }

// #[test]
// #[available_gas(20000000000)]
// fn test_gather_elements2_lastaxis() {
//     let data = data2();
//     let indices = indices22();


//     let y = data.gather_elements(indices: indices,  axis:Option::Some(2));
//     let mut output = y.data;

//     assert(*output[0] == 0, 'not correct');
//     assert(*output[1] == 1, 'not correct');
//     assert(*output[2] == 3, 'not correct');
//     assert(*output[3] == 3, 'not correct');
//     assert(*output[4] == 5, 'not correct');
//     assert(*output[5] == 4, 'not correct');
//     assert(*output[6] == 7, 'not correct');
//     assert(*output[7] == 7, 'not correct');
//     assert(*output[8] == 9, 'not correct');
//     assert(*output[9] == 8, 'not correct');
//     assert(*output[10] == 11, 'not correct');
//     assert(*output[11] == 11, 'not correct');

// }

// #[test]
// #[available_gas(20000000000)]
// fn test_gather_elements2() {
//     let data = data2();
//     let indices = indices22();


//     let y = data.gather_elements(indices: indices,  axis:Option::Some(0));
//     let mut output = y.data;

//     assert(*output[0] == 0, 'not correct');
//     assert(*output[1] == 7, 'not correct');
//     assert(*output[2] == 8, 'not correct');
//     assert(*output[3] == 9, 'not correct');
//     assert(*output[4] == 10, 'not correct');
//     assert(*output[5] == 5, 'not correct');
//     assert(*output[6] == 6, 'not correct');
//     assert(*output[7] == 7, 'not correct');
//     assert(*output[8] == 8, 'not correct');
//     assert(*output[9] == 3, 'not correct');

// }

// #[test]
// #[available_gas(2000000000)]
// fn test_gather_elements_1() {
//     let data = data1();
//     let indices = indices1();


//     let y = data.gather_elements(indices: indices,  axis:Option::Some(0));
//     let mut output = y.data;

//     assert(*output[0] == 1, 'not correct');
//     assert(*output[1] == 2, 'not correct');
//     assert(*output[2] == 1, 'not correct');
//     assert(*output[3] == 3, 'not correct');
//     assert(*output[4] == 3, 'not correct');
// }

// #[test]
// #[available_gas(2000000000)]
// fn test_gather_elements_3() {
//     let data = data();
//     let indices = indices();


//     let y = data.gather_elements(indices: indices,  axis:Option::Some(0));
//     let mut output = y.data;

//     assert(*output[0] == 1, 'not correct');
//     assert(*output[1] == 5, 'not correct');
//     assert(*output[2] == 9, 'not correct');
//     assert(*output[3] == 4, 'not correct');
//     assert(*output[4] == 2, 'not correct');
//     assert(*output[5] == 6, 'not correct');

// }