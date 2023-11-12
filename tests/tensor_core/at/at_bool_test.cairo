// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::ArrayTrait;
    use orion::operators::tensor::BoolTensor;
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


        let mut indices = ArrayTrait::new();
        indices.append(1);

        let result = tensor.at(indices.span());

        assert(result == true, 'result[2] = true');
    }
}


// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::ArrayTrait;
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


        let mut indices = ArrayTrait::new();
        indices.append(1);
        indices.append(1);

        let result = tensor.at(indices.span());

        assert(result == true, 'result[4] = true');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::ArrayTrait;
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


        let mut indices = ArrayTrait::new();
        indices.append(0);
        indices.append(1);
        indices.append(1);

        let result = tensor.at(indices.span());

        assert(result == true, 'result[3] = true');
    }
}
