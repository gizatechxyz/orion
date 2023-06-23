// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::core::{unravel_index};

    #[test]
    #[available_gas(2000000)]
    fn tensor_unravel_index() {
        let mut shape = ArrayTrait::new();
        shape.append(5);
        let result = unravel_index(2, shape.span());
        assert(*result[0] == 2, 'result[0] = 2');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::core::{unravel_index};

    #[test]
    #[available_gas(2000000)]
    fn tensor_unravel_index() {
        let mut shape = ArrayTrait::new();
        shape.append(2);
        shape.append(4);
        let result = unravel_index(6, shape.span());
        assert(*result[0] == 1, 'result[0] = 1');
        assert(*result[1] == 2, 'result[1] = 2');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::core::{unravel_index};


    #[test]
    #[available_gas(2000000)]
    fn tensor_unravel_index() {
        let mut shape = ArrayTrait::new();
        shape.append(2);
        shape.append(4);
        shape.append(6);
        let result = unravel_index(42, shape.span());
        assert(*result[0] == 1, 'result[0] = 1');
        assert(*result[1] == 3, 'result[1] = 3');
        assert(*result[2] == 0, 'result[2] = 0');
    }
}
