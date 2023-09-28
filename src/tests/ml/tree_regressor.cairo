use orion::operators::ml::{FP16x16TreeRegressor, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core::mse;
use orion::numbers::{FP16x16, FixedTrait};

#[test]
#[available_gas(2000000000000)]
fn test_mse() {
    let mut y = array![
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(4, false),
        FixedTrait::new_unscaled(6, false),
        FixedTrait::new_unscaled(8, false)
    ]
        .span();

    let prediction = FixedTrait::<FP16x16>::new_unscaled(5, false);
    let expected_mse = FixedTrait::<FP16x16>::new_unscaled(
        5, false
    ); // MSE = [(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2] / 4 = 5

    let computed_mse = mse(y, prediction);
    assert(computed_mse == expected_mse, 'Failed mse');
}


#[test]
#[available_gas(2000000000000)]
fn test_tree() {
    let data = array![
        array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false)].span(),
        array![FixedTrait::new_unscaled(3, false), FixedTrait::new_unscaled(4, false)].span(),
        array![FixedTrait::new_unscaled(5, false), FixedTrait::new_unscaled(6, false)].span(),
        array![FixedTrait::new_unscaled(7, false), FixedTrait::new_unscaled(8, false)].span(),
    ]
        .span();

    let target = array![
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(4, false),
        FixedTrait::new_unscaled(6, false),
        FixedTrait::new_unscaled(8, false)
    ]
        .span();

    let mut tree = TreeRegressorTrait::fit(data, target, 3, 42);

    let prediction_1 = tree
        .predict(
            array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false),].span()
        );

    let prediction_2 = tree
        .predict(
            array![FixedTrait::new_unscaled(3, false), FixedTrait::new_unscaled(4, false)].span()
        );

    let prediction_3 = tree
        .predict(
            array![FixedTrait::new_unscaled(5, false), FixedTrait::new_unscaled(6, false)].span()
        );

    let prediction_4 = tree
        .predict(
            array![FixedTrait::new_unscaled(7, false), FixedTrait::new_unscaled(8, false)].span()
        );

    assert(prediction_1 == FixedTrait::<FP16x16>::new_unscaled(2, false), 'should predict 2');
    assert(prediction_2 == FixedTrait::<FP16x16>::new_unscaled(4, false), 'should predict 4');
    assert(prediction_3 == FixedTrait::<FP16x16>::new_unscaled(6, false), 'should predict 6');
    assert(prediction_4 == FixedTrait::<FP16x16>::new_unscaled(8, false), 'should predict 8');
}
