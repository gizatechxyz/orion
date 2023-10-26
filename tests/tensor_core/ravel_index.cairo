// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::ArrayTrait;
    use orion::operators::tensor::core::{ravel_index};

    #[test]
    #[available_gas(2000000)]
    fn tensor_ravel_index() {
        let mut shape = ArrayTrait::new();
        shape.append(5);
        let mut indices = ArrayTrait::new();
        indices.append(2);
        let result = ravel_index(shape.span(), indices.span());
        assert(result == 2, 'result = 2');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::ArrayTrait;
    use orion::operators::tensor::core::{ravel_index};

    #[test]
    #[available_gas(2000000)]
    fn tensor_ravel_index() {
        let mut shape = ArrayTrait::new();
        shape.append(2);
        shape.append(4);
        let mut indices = ArrayTrait::new();
        indices.append(1);
        indices.append(2);
        let result = ravel_index(shape.span(), indices.span());
        assert(result == 6, 'result = 6');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::ArrayTrait;
    use orion::operators::tensor::core::{ravel_index};


    #[test]
    #[available_gas(2000000)]
    fn tensor_ravel_index() {
        let mut shape = ArrayTrait::new();
        shape.append(2);
        shape.append(4);
        shape.append(6);
        let mut indices = ArrayTrait::new();
        indices.append(1);
        indices.append(3);
        indices.append(0);
        let result = ravel_index(shape.span(), indices.span());
        assert(result == 42, 'result = 42');
    }
}
