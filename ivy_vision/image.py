"""Collection of image Ivy functions."""

# local
import ivy
import numpy as np
import operator
import functools
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
)
from typing import Union, List, Tuple, Optional


# Extra #
# ------#

@handle_out_argument
@to_native_arrays_and_back
@handle_nestable
def stack_images(
    images: List[Union[ivy.Array, ivy.NativeArray]],
    desired_aspect_ratio: Tuple[int, int] = (1, 1),
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
) -> ivy.Array:
    """Stacks a group of images into a combined windowed image, fitting the desired
    aspect ratio as closely as possible.
    Parameters
    ----------
    images
        Sequence of image arrays to be stacked *[batch_shape,height,width,dims]*
    desired_aspect_ratio:
        desired aspect ratio of the stacked image
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.
    Returns
    -------
    ret
        an array containing the stacked images in a specified aspect ratio/dimensions
    
    """

    return current_backend(images[0]).stack_images(images, desired_aspect_ratio, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def bilinear_resample(
    x: Union[ivy.Array, ivy.NativeArray],
    warp: Union[ivy.Array, ivy.NativeArray],
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
) -> ivy.Array:
    """Performs bilinearly re-sampling on input image.
    Parameters
    ----------
    x
        Input image *[batch_shape,h,w,dims]*.
    warp
        Warp array *[batch_shape,num_samples,2]*
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.
    Returns
    -------
    ret
        Image after bilinear re-sampling.

    """

    return current_backend(x).bilinear_resample(x, warp, out=out)


@to_native_arrays_and_back
@handle_nestable
def gradient_image(
        x: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None

) -> ivy.Array:

    """Computes image gradients (dy, dx) for each channel.
    Parameters
    ----------
    x
        Input image *[batch_shape, h, w, d]* .
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.
    Returns
    -------
    ret
        Gradient images dy *[batch_shape,h,w,d]* and dx *[batch_shape,h,w,d]* .
    
    """
    return current_backend(x).gradient_image(x, out=out)


@to_native_arrays_and_back
@handle_nestable
def float_img_to_uint8_img(
        x: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
) -> ivy.Array:
    """Converts an image of floats into a bit-cast 4-channel image of uint8s, which can
    be saved to disk.
    Parameters
    ----------
    x
        Input float image *[batch_shape,h,w]*.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.
    Returns
    -------
    ret
        The new encoded uint8 image *[batch_shape,h,w,4]* .
    
    """
    x_np = ivy.to_numpy(x).astype("float32")
    x_shape = x_np.shape
    x_bytes = x_np.tobytes()
    x_uint8 = np.frombuffer(x_bytes, np.uint8)
    return ivy.array(np.reshape(x_uint8, list(x_shape) + [4]).tolist(), out=out)


@to_native_arrays_and_back
@handle_nestable
def uint8_img_to_float_img(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
) -> ivy.Array:
    """Converts an image of uint8 values into a bit-cast float image.
    Parameters
    ----------
    x
        Input uint8 image *[batch_shape,h,w,4]*
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.
    Returns
    -------
    ret
        The new float image *[batch_shape,h,w]*
    
    """
    x_np = ivy.to_numpy(x).astype("uint8")
    x_shape = x_np.shape
    x_bytes = x_np.tobytes()
    x_float = np.frombuffer(x_bytes, np.float32)
    return ivy.array(np.reshape(x_float, x_shape[:-1]).tolist(), out=out)


@to_native_arrays_and_back
@handle_nestable
def random_crop(
    x: Union[ivy.Array, ivy.NativeArray],
    crop_size: List[int],
    batch_shape: Optional[List[int]] = None,
    image_dims: Optional[List[int]] = None,
    seed: int = None,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
) -> ivy.Array:
    """Randomly crops the input images.
    Parameters
    ----------
    x
        Input images to crop *[batch_shape,h,w,f]*
    crop_size
        The 2D crop size.
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image dimensions. Inferred from inputs in None. (Default value = None)
    seed
        Required for random number generator
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.
    Returns
    -------
    ret
        The new cropped image *[batch_shape,nh,nw,f]*
    
    """
    x_shape = x.shape
    if batch_shape is None:
        batch_shape = x_shape[:-3]
    if image_dims is None:
        image_dims = x_shape[-3:-1]
    num_channels = x_shape[-1]
    flat_batch_size = functools.reduce(operator.mul, [batch_shape], 1)
    crop_size[0] = min(crop_size[-2], x_shape[-3])
    crop_size[1] = min(crop_size[-1], x_shape[-2])

    # shapes as list
    image_dims = list(image_dims)
    margins = [img_dim - cs for img_dim, cs in zip(image_dims, crop_size)]

    # FBS x H x W x F
    x_flat = ivy.reshape(x, [flat_batch_size] + image_dims + [num_channels])

    # FBS x 1
    rng = np.random.default_rng(seed)
    x_offsets = rng.integers(0, margins[0] + 1, flat_batch_size).tolist()
    y_offsets = rng.integers(0, margins[1] + 1, flat_batch_size).tolist()

    # list of 1 x NH x NW x F
    cropped_list = [
        img[..., xo : xo + crop_size[0], yo : yo + crop_size[1], :]
        for img, xo, yo in zip(ivy.unstack(x_flat, 0, True), x_offsets, y_offsets)
    ]

    # FBS x NH x NW x F
    flat_cropped = ivy.concat(cropped_list, 0)

    # BS x NH x NW x F
    return ivy.reshape(flat_cropped, [batch_shape] + crop_size + [num_channels], out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def linear_resample(
    x: Union[ivy.Array, ivy.NativeArray],
    num_samples: int,
    axis: int = -1,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
) -> Union[ivy.Array, ivy.NativeArray]:
    """Performs linear re-sampling on input image.
    Parameters
    ----------
    x
        Input image
    num_samples
        The number of interpolated samples to take.
    axis
        The axis along which to perform the resample. Default is last dimension.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.
    Returns
    -------
    ret
        The array after the linear resampling.
    
    """
    return current_backend(x).linear_resample(x, num_samples, axis, out=out)