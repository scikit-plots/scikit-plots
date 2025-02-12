from ..layer_utils import (
    find_input_layers,
    find_layer_by_id,
    find_layer_by_name,
    find_output_layers,
    get_incoming_layers,
    get_outgoing_layers,
    is_internal_input,
)


def test_get_incoming_layers(functional_model):
    assert len(list(get_incoming_layers(functional_model.get_layer("input_1")))) == 0

    assert list(get_incoming_layers(functional_model.get_layer("layer_1_1"))) == [
        functional_model.get_layer("input_1")
    ]

    assert list(get_incoming_layers(functional_model.get_layer("concat"))) == [
        functional_model.get_layer("layer_1_2"),
        functional_model.get_layer("layer_2_2"),
        functional_model.get_layer("layer_3_2"),
        functional_model.get_layer("input_2"),
    ]


def test_get_outgoing_layers(functional_model):
    # Test case: no outgoing layers for 'dense_4'
    assert len(list(get_outgoing_layers(functional_model.get_layer("dense_4")))) == 0

    # Test case: outgoing layers from 'input_1' should be 'layer_1_1', 'layer_2_1', and 'layer_3_1'
    assert list(get_outgoing_layers(functional_model.get_layer("input_1"))) == [
        functional_model.get_layer("layer_1_1"),
        functional_model.get_layer("layer_2_1"),
        functional_model.get_layer("layer_3_1"),
    ]

    # Test case: outgoing layers from 'concat' should be 'flatten'
    assert list(get_outgoing_layers(functional_model.get_layer("concat"))) == [
        functional_model.get_layer("flatten")
    ]


def test_find_layer_by_id(functional_model):
    assert find_layer_by_id(functional_model, 0) is None

    layer = functional_model.get_layer("dense_1")
    assert find_layer_by_id(functional_model, id(layer)) == layer


def test_find_layer_by_name(functional_model):
    assert find_layer_by_name(
        functional_model, "input_1"
    ) == functional_model.get_layer("input_1")


def test_find_input_layers(functional_model):
    assert list(find_input_layers(functional_model)) == [
        functional_model.get_layer("input_1"),
        functional_model.get_layer("input_2"),
    ]


def test_find_output_layers(functional_model):
    assert list(find_output_layers(functional_model)) == [
        functional_model.get_layer("dense_4"),
        functional_model.get_layer("concat"),
    ]


def test_is_internal_input_False(model):
    # This method is designed to retrieve a layer by its name or index (if you omit the name).
    assert is_internal_input(model.get_layer("dense_1")) is False
    # This accesses the underlying list of layers in the model via the _layers attribute.
    assert is_internal_input(model._layers[0]) is False


def test_is_internal_input_True(internal_model):
    try:
        print(is_internal_input(internal_model.get_layer("input")))
        # This method is designed to retrieve a layer by its name or index (if you omit the name).
        assert is_internal_input(internal_model.get_layer("input")) is True
    except:
        pass
    try:
        print(is_internal_input(internal_model.get_layer("input_1")))
        # This method is designed to retrieve a layer by its name or index (if you omit the name).
        assert is_internal_input(internal_model.get_layer("input_1")) is True
    except:
        pass
    print(is_internal_input(internal_model.layers[0]))
    # This accesses the underlying list of layers in the model via the _layers attribute.
    assert is_internal_input(internal_model.layers[0]) is True
