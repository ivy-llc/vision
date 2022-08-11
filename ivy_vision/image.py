"""Collection of Image Functions"""

# global
import ivy


def stack_images(images, desired_aspect_ratio):
    """Stacks a group of images into a combined windowed image, fitting the desired
    aspect ratio as closely as possible.
    Parameters
    ----------
    images
        Sequence of image arrays to be stacked *[batch_shape,height,width,dims]*
    desired_aspect_ratio:
        desired aspect ratio of the stacked image
    Returns
    -------
    ret
        an array containing the stacked images in a specified aspect ratio/dimensions

    """
    num_images = len(images)
    if num_images == 0:
        raise Exception("At least 1 image must be provided")
    batch_shape = ivy.shape(images[0])[:-3]
    image_dims = ivy.shape(images[0])[-3:-1]
    num_batch_dims = len(batch_shape)
    if num_images == 1:
        return images[0]
    img_ratio = image_dims[0] / image_dims[1]
    desired_img_ratio = desired_aspect_ratio[0] / desired_aspect_ratio[1]
    stack_ratio = img_ratio * desired_img_ratio
    stack_height = (num_images / stack_ratio) ** 0.5
    stack_height_int = math.ceil(stack_height)
    stack_width_int = math.ceil(num_images / stack_height)
    image_rows = list()
    for i in range(stack_width_int):
        images_to_concat = images[i * stack_height_int : (i + 1) * stack_height_int]
        images_to_concat += [
            ivy.zeros_like(
                images[0], dtype=ivy.dtype(images[0]), device=ivy.dev(images[0])
            )
        ] * (stack_height_int - len(images_to_concat))
        image_rows.append(ivy.concat(images_to_concat, num_batch_dims))

    return ivy.concat(image_rows, axis=num_batch_dims + 1)


def gradient_image(x):

    """Computes image gradients (dy, dx) for each channel.
    Parameters
    ----------
    x
        Input image *[batch_shape, h, w, d]* .

    Returns
    -------
    ret
        Gradient images dy *[batch_shape,h,w,d]* and dx *[batch_shape,h,w,d]* .

    """
    x_shape = ivy.shape(x)
    batch_shape = x_shape[:-3]
    image_dims = x_shape[-3:-1]
    device = ivy.dev(x)
    # to list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)
    num_dims = x_shape[-1]
    # BS x H-1 x W x D
    dy = x[..., 1:, :, :] - x[..., :-1, :, :]
    # BS x H x W-1 x D
    dx = x[..., :, 1:, :] - x[..., :, :-1, :]
    # BS x H x W x D
    dy = ivy.concat(
        (dy, ivy.zeros(batch_shape + [1, image_dims[1], num_dims], device=device)), axis=-3
    )
    dx = ivy.concat(
        (dx, ivy.zeros(batch_shape + [image_dims[0], 1, num_dims], device=device)), axis=-2
    )
    # BS x H x W x D,    BS x H x W x D
    return dx, dy


def float_img_to_uint8_img(x):
    """Converts an image of floats into a bit-cast 4-channel image of uint8s, which can
    be saved to disk.
    Parameters
    ----------
    x
        Input float image *[batch_shape,h,w]*.
    Returns
    -------
    ret
        The new encoded uint8 image *[batch_shape,h,w,4]* .
    """
    x_np = ivy.to_numpy(x).astype("float32")
    x_shape = x_np.shape
    x_bytes = x_np.tobytes()
    x_uint8 = np.frombuffer(x_bytes, np.uint8)
    return ivy.array(np.reshape(x_uint8, list(x_shape) + [4]).tolist())


def uint8_img_to_float_img(x):
    """Converts an image of uint8 values into a bit-cast float image.
    Parameters
    ----------
    x
        Input uint8 image *[batch_shape,h,w,4]*

    Returns
    -------
    ret
        The new float image *[batch_shape,h,w]*

    """
    x_np = ivy.to_numpy(x).astype("uint8")
    x_shape = x_np.shape
    x_bytes = x_np.tobytes()
    x_float = np.frombuffer(x_bytes, np.float32)
    return ivy.array(np.reshape(x_float, x_shape[:-1]).tolist())


def random_crop(x, crop_size, batch_shape=None, image_dims=None, seed: int = None):
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
        for img, xo, yo in zip(ivy.unstack(x_flat, axis=0, keepdims=True), x_offsets, y_offsets)
    ]

    # FBS x NH x NW x F
    flat_cropped = ivy.concat(cropped_list, axis=0)

    # BS x NH x NW x F
    return ivy.reshape(
        flat_cropped, [batch_shape] + crop_size + [num_channels], out=out
    )


def bilinear_resample(x, warp):
    """Performs bilinearly re-sampling on input image.
    Parameters
    ----------
    x
        Input image *[batch_shape,h,w,dims]*.
    warp
        Warp array *[batch_shape,num_samples,2]*

    Returns
    -------
    ret
        Image after bilinear re-sampling.

    """
    batch_shape = x.shape[:-3]
    input_image_dims = x.shape[-3:-1]
    num_feats = x.shape[-1]
    batch_shape = list(batch_shape)
    input_image_dims = list(input_image_dims)
    # image statistics
    height, width = input_image_dims
    max_x = width - 1
    max_y = height - 1
    idx_size = warp.shape[-2]
    batch_shape_flat = int(ivy.prod(ivy.asarray(batch_shape)))
    # B
    batch_offsets = ivy.arange(batch_shape_flat) * height * width
    # B x (HxW)
    base_grid = ivy.tile(ivy.expand_dims(batch_offsets, axis=1), [1, idx_size])
    # (BxHxW)
    base = ivy.reshape(base_grid, [-1])
    # (BxHxW) x D
    data_flat = ivy.reshape(x, [batch_shape_flat * height * width, -1])
    # (BxHxW) x 2
    warp_flat = ivy.reshape(warp, [-1, 2])
    warp_floored = (ivy.floor(warp_flat)).astype(ivy.int32)
    bilinear_weights = warp_flat - ivy.floor(warp_flat)
    # (BxHxW)
    x0 = warp_floored[:, 0]
    x1 = x0 + 1
    y0 = warp_floored[:, 1]
    y1 = y0 + 1
    x0 = ivy.clip(x0, 0, max_x)
    x1 = ivy.clip(x1, 0, max_x)
    y0 = ivy.clip(y0, 0, max_y)
    y1 = ivy.clip(y1, 0, max_y)
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # (BxHxW) x D
    Ia = ivy.gather(data_flat, idx_a, axis=0)
    Ib = ivy.gather(data_flat, idx_b, axis=0)
    Ic = ivy.gather(data_flat, idx_c, axis=0)
    Id = ivy.gather(data_flat, idx_d, axis=0)

    # (BxHxW)
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]
    # (BxHxW) x 1
    wa = ivy.expand_dims((1 - xw) * (1 - yw), axis=1)
    wb = ivy.expand_dims((1 - xw) * yw, axis=1)
    wc = ivy.expand_dims(xw * (1 - yw), axis=1)
    wd = ivy.expand_dims(xw * yw, axis=1)
    # (BxNP) x D
    resampled_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    # B x NP x D
    return ivy.reshape(resampled_flat, batch_shape + [-1, num_feats])
