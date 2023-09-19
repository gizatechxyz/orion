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
use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: TensorTrait::gather docstring
fn gather<
    T, 
    impl TTensorTrait: TensorTrait<T>, 
    impl TCopy: Copy<T>, 
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, indices: Tensor<usize>, axis: Option<usize>
) -> Tensor<T> {
    let data = *self.data;
    let shape = *self.shape;
    let rank = shape.len();

    let axis = match axis {
        Option::Some(val) => val,
        Option::None(_) => 0
    };
    assert (axis < rank, 'axis out of dimensions');

    let rank_indices = indices.shape.len();
    let data_indices = indices.data;
    let axis_shape = *shape.at(axis);
    let ind_max = data_indices.max().unwrap();
    assert(ind_max < axis_shape, 'this index out of bounds');



    
    let mut output_data = ArrayTrait::new();
    let mut output_size = ArrayTrait::new();
    let mut index:usize = 0;
    loop {
        if (index == rank){
            break();
        };
        if (index == axis){
            let mut inner_index: usize = 0;
            loop {
                if (inner_index == rank_indices){
                  break();
                };

                output_size.append(*indices.shape.at(inner_index));
                inner_index +=1;
            }
        }
        else {
            output_size.append(*shape.at(index));
        }
        index += 1;
    };

    let total_elements = data.len();

    let mut outer_loop_break = 1;
    let mut divisor = total_elements;
    let mut ind: usize = 0;
    loop {
        if (ind == axis){
            divisor /= *shape.at(ind);
            break();
        };
        outer_loop_break *= *shape.at(ind);
        divisor /= *shape.at(ind);
        ind +=1;
    };

    let mut break_loop: usize = 1;
    let mut shp = shape;
    let mut ind = rank;
     loop {
        if (ind == axis){
            break();
        };
        break_loop *= *shp.at(ind-1);
        ind -=1;
    };

    let mut outer_loop: usize = 0;
    loop {
        if (outer_loop == outer_loop_break){
            break();
        };

        let mut indices_index =  0;
        loop {
        if (indices_index == data_indices.len()){
                break();
            };
            let indice = *data_indices.at(indices_index);

            let mut inner_loop = 0;
            loop {
            if (inner_loop == break_loop){
                break();
            }; 

            let new_val = inner_loop / divisor % *shape.at(axis);
            if (indice == new_val) {

                let val = break_loop * outer_loop + inner_loop;
                let data_val = *data.at(val);
                output_data.append(data_val);
            }

            inner_loop +=1;
        };

        indices_index +=1;
        };
        
    outer_loop += 1;
        
    };

    let mut output_tensor = TensorTrait::<T>::new(output_size.span(), output_data.span());
  
    return output_tensor;
}
