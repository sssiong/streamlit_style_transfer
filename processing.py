import numpy as np
import PIL.Image
import tensorflow as tf


def bytes_to_tensor(byte_img: bytes) -> np.array:
    max_dim = 512
    img = tf.image.decode_image(byte_img)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def apply_style(base_tensor, style_tensor, model) -> np.array:
    return model(
        tf.constant(base_tensor),
        tf.constant(style_tensor),
    )[0]


def tensor_to_image(tensor: np.array) -> PIL.Image:
    tensor = np.array(tensor * 255, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
