use alexandria_data_structures::array_ext::SpanTraitExt;
use core::array::ArrayTrait;
use core::array::SpanTrait;

use core::traits::Into;
use core::debug::PrintTrait;
use core::traits::TryInto;
use core::serde::Serde;
use core::traits::Destruct;
use core::option::OptionTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use core::dict::Felt252DictTrait;
use core::nullable::{nullable_from_box, match_nullable, FromNullableResult};
/// Cf: TensorTrait::scatter_nd docstring
fn scatter_nd<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TAdd: Add<T>,
    impl TMul: Mul<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
>(
    self: @Tensor<T>,
    updates: Tensor<T>,
    indices: Tensor<usize>,
    reduction: Option<usize>
) -> Tensor<T> {
   
    let reduction = match reduction {
        Option::Some(val) => val,
        Option::None(_) => 'none'
    };

    let data_rank = (*self.shape).len();
    let indices_rank = (indices.shape).len();
    let updates_rank = (updates.shape).len();
    let mut data_shape = *self.shape;
    let mut indices_shape = indices.shape;
    let updates_shape = updates.shape;

    let indices_last_axis = indices_shape.pop_back().unwrap();
    assert(*indices_last_axis <= data_rank, 'must be <= data rank');

    let ind_max = indices.data.max().unwrap();
    if (data_rank > 1){
        assert(ind_max < data_rank, 'index is out of bound');
    }

    let mut batch_dims_shape = ArrayTrait::new();
    let mut ind: usize = 0;

    loop {
        match indices_shape.pop_front() {
            Option::Some(val) => { batch_dims_shape.append(*val);},
            Option::None(_) => { break; }
        };
    };

    let mut data_shape_clone = data_shape.clone();
    loop {
        match data_shape_clone.pop_front() {
            Option::Some(val) => { 
                if (ind >= *indices_last_axis) {
                    batch_dims_shape.append(*val);
                    }
                },
            Option::None(_) => { break; }
        };
    };

    let mut ind: usize = 0;
    loop {
        match batch_dims_shape.pop_front() {
        Option::Some(val) => { 
            assert(val == *updates_shape[ind], 'must be same');
            },
        Option::None(_) => { break; }
        };
    };

    let mut data_indices = indices.data;
    let mut data_updates = updates.data;

    let mut data_shape_clone = data_shape.clone();
    let mut indexer = 1;
    let data_shape_first = data_shape_clone.pop_front();
    if data_rank >= 1 {
        loop {
            match data_shape_clone.pop_front() {
                Option::Some(val) => { indexer *= *val;},
                Option::None(_) => { break; }
            };
        }
    }

    let mut updates_index_dict: Felt252Dict<u32> = Default::default();
    let mut dict_ind: usize = 1;
    loop {
        match data_indices.pop_front() {
            Option::Some(val) => { 
                updates_index_dict.insert((*val).into(), dict_ind);
                dict_ind += 1;
            },
            Option::None(_) => { break; }
        };
    };


    let mut output_data = ArrayTrait::<T>::new();
    let mut data = *self.data;
    let mut index: usize = 0;
    let mut inner_index: usize = 0;

    let num = *data_shape_first.unwrap();
    loop {
        if (index == num){
            break;
        }
        let updates_index = (index/indexer);
        let comp_index = updates_index_dict.get(index.into());

        if (comp_index == 0) {
            loop {
                if (inner_index == indexer) { 
                    inner_index = 0;
                    break; 
                }
                let val = *data.at((index * indexer) + inner_index);
                output_data.append(val);
                inner_index += 1;
            };
        }  

        else {
            loop {
                if (inner_index == indexer) { 
                    inner_index = 0;
                    break; 
                }
                if (reduction == 'none'){
                    let val = data_updates.at(((comp_index-1) * indexer) + inner_index);
                    output_data.append(*val);
                }
                if (reduction == 'add') {
                    let val = data_updates.at(((comp_index-1) * indexer) + inner_index);
                    let data_val = *data.at((index * indexer) + inner_index);
                    output_data.append(*val + data_val);
                }

                if (reduction == 'mul') {
                    let val = data_updates.at(((comp_index-1) * indexer) + inner_index);
                    let data_val = *data.at((index * indexer) + inner_index);
                    output_data.append((*val) * data_val);
                }

                if (reduction == 'max') {
                    let val = data_updates.at(((comp_index-1) * indexer) + inner_index);
                    let data_val = *data.at((index * indexer) + inner_index);
                    if (*val > data_val) {
                        output_data.append(*val);
                    }
                    else {
                        output_data.append(data_val);
                    }
                 }

                if (reduction == 'min') {
                    let val = data_updates.at(((comp_index-1) * indexer) + inner_index);
                    let data_val = *data.at((index * indexer) + inner_index);
                    if (*val > data_val) {
                        output_data.append(data_val);
                    }
                    else {
                        output_data.append(*val);
                    }
                 }
                

                inner_index += 1;
            }
        }
        index += 1;
       
    };


    let mut output_tensor = TensorTrait::<T>::new(*self.shape, output_data.span());
    // let mut output_tensor = TensorTrait::<T>::new(*self.shape, *self.data);
    return output_tensor;

}










// Tests--------------------------------------------------------------------------------------------------------------

use orion::utils::assert_eq;

fn indices() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(2);

    TensorTrait::new(shape.span(), data.span())
}

fn data() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(4);
    sizes.append(4);
    sizes.append(4);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);

    data.append(8);
    data.append(7);
    data.append(6);
    data.append(5);
    data.append(4);
    data.append(3);
    data.append(2);
    data.append(1);

    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);

    data.append(8);
    data.append(7);
    data.append(6);
    data.append(5);
    data.append(4);
    data.append(3);
    data.append(2);
    data.append(1);

    data.append(8);
    data.append(7);
    data.append(6);
    data.append(5);
    data.append(4);
    data.append(3);
    data.append(2);
    data.append(1);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);

    data.append(8);
    data.append(7);
    data.append(6);
    data.append(5);
    data.append(4);
    data.append(3);
    data.append(2);
    data.append(1);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);


    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn updates() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(4);
    sizes.append(4);

    let mut data = ArrayTrait::new();
    data.append(5);
    data.append(5);
    data.append(5);
    data.append(5);

    data.append(6);
    data.append(6);
    data.append(6);
    data.append(6);
    data.append(7);
    data.append(7);
    data.append(7);
    data.append(7);
    data.append(8);
    data.append(8);
    data.append(8);
    data.append(8);

    data.append(1);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(2);
    data.append(2);
    data.append(2);
    data.append(2);
    data.append(3);
    data.append(3);
    data.append(3);
    data.append(3);
    data.append(4);
    data.append(4);
    data.append(4);
    data.append(4);

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn data2() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(8);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);

    TensorTrait::new(shape.span(), data.span())
}

fn indices2() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(4);
    data.append(3);
    data.append(1);
    data.append(7);

    TensorTrait::new(shape.span(), data.span())
}

fn updates2() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(9);
    data.append(10);
    data.append(11);
    data.append(12);

    TensorTrait::new(shape.span(), data.span())
}

#[test]
#[available_gas(20000000000)]
fn test_scatter_default() {
    let data = data();
    let indices = indices();
    let updates = updates();

    // let y = data.scatter_nd(updates:updates, indices: indices,  reduction:Option::None(()));
    let y = data.scatter_nd(updates:updates, indices: indices,  reduction:Option::Some('add'));
    let mut output = y.data;

    // loop {
    //     match output.pop_front() {
    //         Option::Some(val) => {
    //            (*val).print();
    //         },
    //         Option::None(_) => { break; }
    //     };
    // };

}

// #[test]
// #[available_gas(20000000000)]
// fn test_scatter_nd_example() {
//     let tensor = TensorTrait::<u32>::new(
//         shape: array![4, 4, 4].span(), 
//         data: array![1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6,
//        7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4,
//        5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8].span()
//     );

//     let updates = TensorTrait::<u32>::new(
//             shape: array![2, 4, 4].span(), 
//             data: array![5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 1, 1, 1, 1, 2, 2,
//                     2, 2, 3, 3, 3, 3, 4, 4, 4, 4].span(), 
//     );

//     let indices = TensorTrait::<u32>::new(
//             shape: array![2, 1].span(), 
//             data: array![0, 2].span(), 
//     );
    
//     let y = tensor.scatter_nd(updates:updates, indices: indices,  reduction:Option::Some('add'));
//     let mut output = y.data;

//     loop {
//         match output.pop_front() {
//             Option::Some(val) => {
//                (*val).print();
//             },
//             Option::None(_) => { break; }
//         };
//     };
// }