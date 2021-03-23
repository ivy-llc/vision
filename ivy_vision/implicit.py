# global
import ivy


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
    cumprod_one_m_occ_prob = ivy.cumprod(1 - occ_prob, -1)
    return ivy.concatenate((occ_prob[..., 0:1], occ_prob[..., 1:] * cumprod_one_m_occ_prob[..., :-1]), -1)


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

    # BS_flat
    bin_sizes_flat = ivy.reshape(bin_sizes, (-1,))

    # BS_flat x NS
    random_offsets_flat = ivy.concatenate(
        [ivy.random_uniform(0, bin_size, shape=(1, num_samples))
         for bin_size in ivy.unstack(bin_sizes_flat, 0)], 0)

    # BS x NS
    random_offsets = ivy.reshape(random_offsets_flat, batch_shape + [num_samples])

    # BS x NS
    return linspace_vals + random_offsets


# noinspection PyUnresolvedReferences
def render_rays_via_termination_probabilities(radial_depths, features, densities, render_variance=False):
    """
    Render features onto the image plane, given rays sampled at radial depths with readings of
    feature values and densities at these sample points.

    :param radial_depths: Radial depth values *[batch_shape,num_samples_per_ray+1]*
    :type radial_depths: array
    :param features: Feature values at the sample points *[batch_shape,num_samples_per_ray,feat_dim]*
    :type features: array
    :param densities: Volume density values at the sample points *[batch_shape,num_samples_per_ray]*
    :type densities: array
    :param render_variance: Whether to also render the feature variance. Default is False.
    :type render_variance: bool, optional
    :return: The feature renderings along the rays, computed via the termination probabilities *[batch_shape,feat_dim]*
    """

    # BS x NSPR
    d = radial_depths[..., 1:] - radial_depths[..., :-1]
    ray_term_probs = ray_termination_probabilities(densities, d)
    rendering = ivy.reduce_sum(ray_term_probs * features, -2)
    if not render_variance:
        return rendering
    var = ivy.reduce_sum(ray_term_probs*(rendering - features)**2, -2)
    return rendering, var


# noinspection PyUnresolvedReferences
def render_ray_variances_via_termination_probabilities(radial_depths, features, densities):
    """
    Render features onto the image plane, given rays sampled at radial depths with readings of
    feature values and densities at these sample points.

    :param radial_depths: Radial depth values *[batch_shape,num_samples_per_ray+1]*
    :type radial_depths: array
    :param features: Feature values at the sample points *[batch_shape,num_samples_per_ray,feat_dim]*
    :type features: array
    :param densities: Volume density values at the sample points *[batch_shape,num_samples_per_ray]*
    :type densities: array
    :return: The feature renderings along the rays, computed via the termination probabilities *[batch_shape,feat_dim]*
    """

    # BS x NSPR
    rendering = render_rays_via_termination_probabilities(radial_depths, features, densities)
    d = radial_depths[..., 1:] - radial_depths[..., :-1]
    ray_term_probs = ray_termination_probabilities(densities, d)
    return ivy.reduce_sum(ray_term_probs * features, -2)


# noinspection PyUnresolvedReferences
def render_rays_via_quadrature_rule(radial_depths, features, densities):
    """
    Render features onto the image plane, given rays sampled at radial depths with readings of
    feature values and densities at these sample points.

    :param radial_depths: Radial depth values *[batch_shape,num_samples_per_ray+1]*
    :type radial_depths: array
    :param features: Feature values at the sample points *[batch_shape,num_samples_per_ray,feat_dim]*
    :type features: array
    :param densities: Volume density values at the sample points *[batch_shape,num_samples_per_ray]*
    :type densities: array
    :return: The feature renderings along the rays, computed via the quadrature rule *[batch_shape,feat_dim]*
    """

    # BS x NSPR
    d = radial_depths[..., 1:] - radial_depths[..., :-1]

    # BS x NSPR x 1
    T = ivy.expand_dims(ivy.exp(-ivy.cumsum(densities * d, -1)), -1)

    # BS x FD
    return ivy.reduce_sum(T * ivy.expand_dims(1-ivy.exp(-densities*d), -1) * features, -2)
