import tensorflow as tf
import os


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)

    return image

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image


def random_crop(image, img_height, img_width):
  cropped_image = tf.image.random_crop(
      image, size=[img_height, img_width, 3])

  return cropped_image


def normalize(input_image):
    image = tf.cast(input_image, tf.float32)
    input_image = (input_image / 127.5) - 1


    return input_image


def random_jitter(image, img_height, img_width):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image, img_height, img_width)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image
