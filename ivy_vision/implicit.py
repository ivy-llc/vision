# global
import ivy


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
