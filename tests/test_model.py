from quad_pendulum.model import make_cart_pendulum_xml


def test_xml_contains_requested_number_of_hinges():
    xml = make_cart_pendulum_xml(links=4)

    assert xml.count('type="hinge"') == 4
    assert 'name="cart_force"' in xml
    assert 'frictionloss="0.001"' in xml


def test_xml_rejects_invalid_link_count():
    try:
        make_cart_pendulum_xml(links=0)
    except ValueError as exc:
        assert "links must be between 1 and 4" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
