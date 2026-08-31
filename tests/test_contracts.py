from nb_vision.contracts import CLASS_NAMES, detection_packet, validate_yolo_line


def test_class_contract_is_stable():
    assert CLASS_NAMES == ("mouse", "bottle")


def test_valid_yolo_line_is_normalized_and_parsed():
    assert validate_yolo_line("1 0.5 0.2 0.1 0.3") == (1, 0.5, 0.2, 0.1, 0.3)


def test_invalid_yolo_line_is_rejected():
    try:
        validate_yolo_line("2 0.5 0.2 0.1 0.3")
    except ValueError as error:
        assert "class id" in str(error)
    else:
        raise AssertionError("invalid class must fail")


def test_detection_packet_has_required_fields():
    assert detection_packet(
        12.44,
        [{"name": "mouse", "score": 0.91, "xyxy": [1, 2, 3, 4]}],
    ) == ('{"speed_fps": 12.4, "objects": [{"name": "mouse", "score": 0.91, "xyxy": [1, 2, 3, 4]}]}')
