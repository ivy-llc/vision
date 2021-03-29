# global
import ivy


def sinusoid_positional_encoding(x, embedding_length=10):
    """
    Perform sinusoid positional encoding of the inputs.

    :param x: input array to encode *[batch_shape, dim]*
    :type x: array
    :param embedding_length: Length of the embedding. Default is 10.
    :type embedding_length: int, optional
    :return: The new positionally encoded array *[batch_shape, dim+dim*2*embedding_length]*
    """
    rets = [x]
    for i in range(embedding_length):
        for fn in [ivy.sin, ivy.cos]:
            rets.append(fn(2.**i * x))
    return ivy.concatenate(rets, -1)


# noinspection PyUnresolvedReferences
def sampled_volume_density_to_occupancy_probability(density, inter_sample_distance):
    """
    Compute probability of occupancy, given sampled volume densities and their associated inter-sample distances

    :param density: The sampled density values *[batch_shape]*
    :type density: array
    :param inter_sample_distance: The inter-sample distances *[batch_shape]*
    :type inter_sample_distance: array
    :return: The occupancy probabilities *[batch_shape]*
    """
    return 1 - ivy.exp(-density*inter_sample_distance)


def ray_termination_probabilities(density, inter_sample_distance):
    """
    Compute probability of occupancy, given sampled volume densities and their associated inter-sample distances

    :param density: The sampled density values *[batch_shape,num_samples_per_ray]*
    :type density: array
    :param inter_sample_distance: The inter-sample distances *[batch_shape,num_samples_per_ray]*
    :type inter_sample_distance: array
    :return: The occupancy probabilities *[batch_shape]*
    """

    # BS x NSPR
    occ_prob = sampled_volume_density_to_occupancy_probability(density, inter_sample_distance)
    return occ_prob * ivy.cumprod(1. - occ_prob + 1e-10, -1, exclusive=True)


# noinspection PyUnresolvedReferences
def stratified_sample(starts, ends, num_samples, batch_shape=None):
    """
    Perform stratified sampling, between start and end arrays. This operation divides the range into equidistant bins,
    and uniformly samples value within the ranges for each of these bins.

    :param starts: Start values *[batch_shape]*
    :type starts: array
    :param ends: End values *[batch_shape]*
    :type ends: array
    :param num_samples: The number of samples to generate between starts and ends
    :type num_samples: int
    :param batch_shape: Shape of batch, Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :return: The stratified samples, with each randomly placed in uniformly spaced bins *[batch_shape,num_samples]*
    """

    # shapes
    if batch_shape is None:
        batch_shape = starts.shape

    # shapes as lists
    batch_shape = list(batch_shape)

    # BS
    bin_sizes = (ends - starts)/num_samples

    # BS x NS
    linspace_vals = ivy.linspace(starts, ends - bin_sizes, num_samples)

    # BS x NS
    random_uniform = ivy.random_uniform(shape=batch_shape + [num_samples], dev_str=ivy.dev_str(starts))

    # BS x NS
    random_offsets = random_uniform * ivy.expand_dims(bin_sizes, -1)

    # BS x NS
    return linspace_vals + random_offsets


# noinspection PyUnresolvedReferences
def render_rays_via_termination_probabilities(ray_term_probs, features, render_variance=False):
    """
    Render features onto the image plane, given rays sampled at radial depths with readings of
    feature values and densities at these sample points.

    :param ray_term_probs: The ray termination probabilities *[batch_shape,num_samples_per_ray]*
    :type ray_term_probs: array
    :param features: Feature values at the sample points *[batch_shape,num_samples_per_ray,feat_dim]*
    :type features: array
    :param render_variance: Whether to also render the feature variance. Default is False.
    :type render_variance: bool, optional
    :return: The feature renderings along the rays, computed via the termination probabilities *[batch_shape,feat_dim]*
    """

    # BS x NSPR
    rendering = ivy.reduce_sum(ray_term_probs * features, -2)
    if not render_variance:
        return rendering
    var = ivy.reduce_sum(ray_term_probs*(ivy.expand_dims(rendering, -2) - features)**2, -2)
    return rendering, var


def render_implicit_features_and_depth(network_fn, rays_o, rays_d, near, far, num_samples, render_variance=False,
                                       v=None):
    """
    Render an rgb-d image, given an implicit rgb and density function conditioned on xyz data.

    :param network_fn: the implicit function.
    :type network_fn: callable
    :param rays_o: The camera center *[batch_shape,feat]*
    :type rays_o: array
    :param rays_d: The rays in world space *[batch_shape,ray_batch_shape,feat]*
    :type rays_d: array
    :param near: The near clipping plane values *[batch_shape,ray_batch_shape]*
    :type near: array
    :param far: The far clipping plane values *[batch_shape,ray_batch_shape]*
    :type far: array
    :param num_samples: The number of stratified samples to use along each ray
    :type num_samples: int
    :param render_variance: Whether to also render the feature variance. Default is False.
    :type render_variance: bool, optional
    :param v: The container of trainable variables for the implicit model. default is to use internal variables.
    :type v: ivy Container of variables
    :return: The rendered feature *[batch_shape,ray_batch_shape,feat]* and depth *[batch_shape,ray_batch_shape,1]* values
    """

    # Compute 3D query points
    z_vals = ivy.expand_dims(stratified_sample(near, far, num_samples), -1)
    rays_d = ivy.expand_dims(rays_d, -2)
    rays_o = ivy.broadcast_to(rays_o, rays_d.shape)
    pts = rays_o + rays_d * z_vals

    # Run network
    feat, densities = network_fn(pts, v=v)

    z_vals_w_terminal = ivy.concatenate((z_vals[..., 0], ivy.ones_like(z_vals[..., -1:, 0])*1e10), -1)
    depth_diffs = z_vals_w_terminal[..., 1:] - z_vals_w_terminal[..., :-1]
    ray_term_probs = ivy.expand_dims(ray_termination_probabilities(densities, depth_diffs), -1)
    feat = render_rays_via_termination_probabilities(ray_term_probs, feat, render_variance)
    depth = render_rays_via_termination_probabilities(ray_term_probs, z_vals, render_variance)
    if render_variance:
        return (*feat, *depth)
    return feat, depth
