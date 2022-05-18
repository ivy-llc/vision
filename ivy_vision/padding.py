"""Collection of Padding Functions"""

# global
import ivy as _ivy


def pad_omni_image(image, pad_size, image_dims=None):
    """Pad an omni-directional image with the correct image wrapping at the edges.

    Parameters
    ----------
    image
        Image to perform the padding on *[batch_shape,h,w,d]*
    pad_size
        Number of pixels to pad.
    image_dims
        Image dimensions. Inferred from Inputs if None. (Default value = None)

    Returns
    -------
    ret
        New padded omni-directional image *[batch_shape,h+ps,w+ps,d]*

    """

    if image_dims is None:
        image_dims = image.shape[-3:-1]

    # BS x PS x W/2 x D
    top_left = image[..., 0:pad_size, int(image_dims[1] / 2):, :]
    top_right = image[..., 0:pad_size, 0:int(image_dims[1] / 2), :]

    # BS x PS x W x D
    top_border = _ivy.flip(_ivy.concatenate((top_left, top_right), -2), -3)

    # BS x PS x W/2 x D
    bottom_left = image[..., -pad_size:, int(image_dims[1] / 2):, :]
    bottom_right = image[..., -pad_size:, 0:int(image_dims[1] / 2), :]

    # BS x PS x W x D
    bottom_border = _ivy.flip(_ivy.concatenate((bottom_left, bottom_right), -2), -3)

    # BS x H+2PS x W x D
    image_expanded = _ivy.concatenate((top_border, image, bottom_border), -3)

    # BS x H+2PS x PS x D
    left_border = image_expanded[..., -pad_size:, :]
    right_border = image_expanded[..., 0:pad_size, :]

    # BS x H+2PS x W+2PS x D
    return _ivy.concatenate((left_border, image_expanded, right_border), -2)
