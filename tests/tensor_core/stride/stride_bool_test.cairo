// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use core::array::ArrayTrait;
    use orion::operators::tensor::{BoolTensor};
    use orion::operators::tensor::core::{TensorTrait};


    #[test]
    #[available_gas(2000000)]
    fn tensor_at() {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);

        let mut data = ArrayTrait::new();
        data.append(false);
        data.append(true);
        data.append(false);

        let tensor = TensorTrait::<bool>::new(sizes.span(), data.span());

        let result = tensor.stride();

        assert(*result[0] == 1, 'stride x = 1');
        assert(result.len() == 1, 'len = 1');
    }
}


// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use core::array::ArrayTrait;
    use orion::operators::tensor::{BoolTensor};
    use orion::operators::tensor::core::{TensorTrait};


    #[test]
    #[available_gas(2000000)]
    fn tensor_at() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(false);
        data.append(false);
        data.append(false);
        data.append(true);

        let tensor = TensorTrait::<bool>::new(sizes.span(), data.span());

        let result = tensor.stride();

        assert(*result[0] == 2, 'stride x = 2');
        assert(*result[1] == 1, 'stride y = 1');
        assert(result.len() == 2, 'len = 2');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use core::array::ArrayTrait;
    use orion::operators::tensor::{BoolTensor};
    use orion::operators::tensor::core::{TensorTrait};


    #[test]
    #[available_gas(2000000)]
    fn tensor_at() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(false);
        data.append(false);
        data.append(false);
        data.append(true);
        data.append(false);
        data.append(false);
        data.append(false);
        data.append(false);

        let tensor = TensorTrait::<bool>::new(sizes.span(), data.span());

        let result = tensor.stride();

        assert(*result[0] == 4, 'stride x = 4');
        assert(*result[1] == 2, 'stride y = 2');
        assert(*result[2] == 1, 'stride z = 1');
        assert(result.len() == 3, 'len = 3');
    }
}
