import numpy as np

from app.preprocess import crop_black, crop_resize, preprocess


def test_crop_black_removes_black_border_and_resize_is_deterministic():
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    image[20:100, 30:150] = np.arange(120, dtype=np.uint8)[None, :, None] + 30

    cropped = crop_black(image)
    first = crop_resize(image, 224)
    second = crop_resize(image, 224)

    assert cropped.shape == (80, 120, 3)
    assert cropped.dtype == np.uint8
    assert first.shape == (224, 224, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)


def test_preprocess_is_float32_rgb_and_batched():
    image = np.full((100, 200, 3), 100, dtype=np.uint8)
    output = preprocess(image, 224)
    assert output.shape == (1, 224, 224, 3)
    assert output.dtype.name == "float32"
