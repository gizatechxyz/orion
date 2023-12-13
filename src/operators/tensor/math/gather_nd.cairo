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
fn gather_nd<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, indices: Tensor<usize>, batch_dims: Option<usize>
) -> Tensor<T> {
    let batch_dims = match batch_dims {
        Option::Some(val) => val,
        Option::None(_) => 0
    };

    let data_rank = (*self.shape).len();
    let indices_rank = (indices.shape).len();
    assert((data_rank >= 1 ) & (indices_rank >= 1), 'rank must > 1');
   
    let mut data_shape = *self.shape;
    let mut indices_shape = indices.shape;
    let mut data_shape_clone = data_shape.clone();
    let mut indices_shape_clone = indices_shape.clone();

    let indices_shape_last = indices_shape_clone.pop_back().unwrap();
    assert((*indices_shape_last >= 1) & (*indices_shape_last <= data_rank-batch_dims), 'check indices');

    let mut batch_dims_shape = ArrayTrait::new();
    let mut output_shape = ArrayTrait::new();
    let mut index_data =  ArrayTrait::new();
    let mut output_data = ArrayTrait::new();

    let mut batch_dims_size = batch_dims;
    let mut total_data_len = 1;
    let mut multiple_data_len = ArrayTrait::new();

    let mut ind = 0;
    loop {
        if (ind == batch_dims) {
            break();
        }
        match indices_shape_clone.pop_front() {
            Option::Some(val) => {
                batch_dims_size *= *val;
                batch_dims_shape.append(*val);
                ind += 1;
            },
            Option::None(_) => { break; }
        };
    };

    loop {
        match indices_shape_clone.pop_front() {
            Option::Some(val) => {
                batch_dims_shape.append(*val);
            },
            Option::None(_) => { break; }
        };
    };

    if (*indices_shape_last == data_rank - batch_dims) {
        output_shape = batch_dims_shape;
    }
    else {
        let mut ind = 0;
        let mut multiple = 1;
        output_shape = batch_dims_shape;
        loop {
            match data_shape_clone.pop_front() {
                Option::Some(val) => {
                    if (ind >= (batch_dims + *indices_shape_last)) {
                        output_shape.append(*val);
                    }
                    // total_data_len *= *val;
                    ind += 1;
                },
                Option::None(_) => { break; }
            };
        };
    }

    let mut ind = 0;
    let mut multiple = 1;
    let mut data_shape_clone = data_shape.clone();
    loop {
        match data_shape_clone.pop_front() {
            Option::Some(val) => {
                if (ind >= batch_dims) {
                    multiple *= *val;
                    multiple_data_len.append(multiple);
                }
                ind += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let mut ind = 0;
    let mut incrementer = 1;
    let mut data_shape_clone = data_shape.clone();
    loop {
        match data_shape_clone.pop_front() {
            Option::Some(val) => {
                if (ind >= batch_dims + *indices_shape_last) {
                    incrementer *= *val;
                }
                ind += 1;
            },
            Option::None(_) => { break; }
        };
    };


    let mut ind = 0;
    let mut indices_shape_clone = indices_shape.clone();
    let mut breaker = 1;
    loop {
        match indices_shape_clone.pop_front() {
            Option::Some(val) => {
                if (ind >= batch_dims) {
                    breaker *= *val;
                }
                ind += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let data_shape_last = data_shape.pop_back().unwrap();

    total_data_len = *multiple_data_len.at(multiple_data_len.len() - 1);
    let mut data_indices = indices.data;
    let mut ind = 0;
    let mut result = 0;
    let mut outer_loop = 0;

    loop {
        match data_indices.pop_front() {
            Option::Some(val) => {
                let index = ind % *indices_shape_last;
                let incr=  total_data_len * (ind / breaker);
                result += (*val * total_data_len / *multiple_data_len.at(index));
                ind += 1;

                if (index == *indices_shape_last-1) {
                    let mut data_ind:usize = result ;
                    loop {
                        if data_ind == result + incrementer { break; }
                        index_data.append(data_ind + incr);
                        data_ind+=1;
                    };
                    result = 0;
                };
            },
            Option::None(_) => { break; }
        };
    };


    loop {
        match index_data.pop_front() {
            Option::Some(val) => {
               output_data.append(*self.data[val]);
            },
            Option::None(_) => { break; }
        };
    };

    let mut output_tensor = TensorTrait::<T>::new(output_shape.span(), output_data.span());

    // let mut output_tensor = TensorTrait::<T>::new(*self.shape, *self.data);
    return output_tensor;
}


// Tests--------------------------------------------------------------------------------------------------------------


use orion::utils::assert_eq;


fn indices() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(4);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(1);
    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn indices1() -> Tensor<u32> {
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

fn data() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn data1() -> Tensor<u32> {
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

fn u32_tensor_3x3x3_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);
    sizes.append(2);
    sizes.append(4);

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
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);
    data.append(26);

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn indices_333() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);
    sizes.append(1);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(0);
    data.append(0);
    
    data.append(0);
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

    data.append(0);
    data.append(1);
    data.append(1);

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}


#[test]
#[available_gas(20000000000)]
fn test_gather_elements_default() {
    let data = u32_tensor_3x3x3_helper();
    let indices = indices_333();

    let y = data.gather_nd(indices: indices,  batch_dims:Option::Some(1));
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