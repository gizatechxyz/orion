use orion::operators::tensor::{Tensor, TensorTrait, U32Tensor};
use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;
use orion::operators::matrix::{MutMatrixTrait, MutMatrix, MutMatrixImpl};
use orion::operators::sequence::SequenceTrait;


/// Cf: TensorTrait::split_to_sequence docstring
fn split_to_sequence<
    T,
    +Copy<T>,
    +Drop<T>,
    +TensorTrait<T>,
    +SequenceTrait<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +PartialEq<Tensor<T>>,
    +PartialOrd<Tensor<T>>
>(
self: @Tensor<T>, split: Option<Tensor<usize>>, axis:usize, keepdims:Option<bool> ) -> Array<Tensor<T>> {

    let has_split = match split {
        Option::Some(value) => {
            true
        },
        Option::None => false,
    };

    let rank = (*self).shape.len();
    // assert(axis < rank && axis > -rank, 'axis out of dimensions');
    assert(axis < rank, 'axis out of dimensions');

     if has_split ==true {
          let split_shape_len = split.unwrap().shape.len();
          assert(split_shape_len <= 1, 'split shape is invalid');
     }


    let mut split_length: Array<usize> = array![];
    let mut i: usize = 0;
    if has_split ==false {
        loop {
            if (i>=*(*self).shape.at(axis)) {
                break;
            }
            split_length.append(1);
            i += 1;
        };
    }
    //scalar
    else if split.unwrap().shape.len()==0 {
        let mut dim = *(*self).shape.at(axis);
        let length = split.unwrap().data.at(0);
        let mut n = dim/*length;
        let mut i: usize = 0;
        loop {
            if i>=n {
                break;
            }
            split_length.append(*length);
            i += 1;
        };
        
        let mut left = dim - *length * n;

        if left > 0 {
            split_length.append(left);    
        }
    }
    else {
        let mut tmp_split = split.unwrap();
        loop {
        let mut i: usize = 0;
        match tmp_split.data.pop_front() { 
            Option::Some(item) => { 
                split_length.append(*item);
                i += 1;
            },
            Option::None(_) => { break; }
        };
    };
    }


    let mut splited_t: Array<Tensor<T>> = array![];
    let mut sli: MutMatrix<usize> =  MutMatrixImpl::new((*self).shape.len(), 2);  
    let mut pos: usize = 0;
    let mut i = 0;

    let mut tmp_tensor = (*self);
    loop {
        match tmp_tensor.shape.pop_front() {
            Option::Some(item) => { 
                let s: usize = *item;
                sli.set(i,0,0); 
                sli.set(i,1,s); 
                i += 1;
            },
            Option::None(_) => { break; }
        };
    };



    loop {
    let mut i: usize = 0;
    let mut ele: usize = 0;
    match split_length.pop_front() {
    Option::Some(item) => { 
            let mut spl: usize = item;
            sli.set(axis, 0, pos);
            pos += spl; 
            sli.set(axis, 1, pos);

            if rank<2{
                ele =axis
            }
            
            let last_ele_0  = match sli.get(1,0) {
                        Option::Some(res) => {
                            res
                        },
                        Option::None(_) => {
                            assert(false, 'Failed to fetch last_ele_0');
                            0
                        },
            };
            let last_ele_1 = match sli.get(1, 1) {
                        Option::Some(res) => {
                            res
                        },
                        Option::None(_) => {
                            assert(false, 'Failed to fetch last_ele_1');
                            0
                        },
            };
            let starts: Span<usize> = array![sli.get(0,0).unwrap(),last_ele_0].span();
            let ends: Span<usize> = array![ sli.get(0,1).unwrap(), last_ele_1].span();
            let axes: Option<Span<usize>> = Option::None(());
            let steps: Option<Span<usize>> = Option::None(());
            let mut sub_t: Tensor<T> = self.slice(starts, ends, axes, steps);
            let mut len = sub_t.shape.len();
            splited_t.append(sub_t);
            i += 1;
    },
    Option::None(_) => { break; }
    };
};

    
    let mut seq_result = SequenceTrait::sequence_construct(splited_t);

    let keepdims = match split {
        Option::Some(value) => {
            true
        },
        Option::None => false,
    };

    if has_split == false && keepdims == false { 
        let mut res: Array<Tensor<T>> = array![];
        let mut i: usize = 0;
        loop {
        match seq_result.pop_front() {
            Option::Some(item) => { 
                let mut reshaped_tensor = item.squeeze(axes: Option::None(()));
                res.append(reshaped_tensor);
                i+=1;
            },
            Option::None(_) => { break; }
        };
    };
     seq_result = SequenceTrait::sequence_construct(res);
    
    };

    return seq_result;

}