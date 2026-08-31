from nb_vision.prelabel import coco_to_project_class


def test_maps_coco_mouse_and_bottle_only():
    assert coco_to_project_class("mouse") == 0
    assert coco_to_project_class("bottle") == 1
    assert coco_to_project_class("cup") is None
