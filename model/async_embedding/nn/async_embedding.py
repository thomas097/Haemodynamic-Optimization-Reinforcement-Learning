import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


class AsyncFeatureExtraction(torch.nn.Module):
    def __init__(self, in_channels, out_channels, out_positions, d_embedding=8, kernel_smoothing=1):
        """ Computes from an irregularly-sampled time series of asynchronous
        features a new series with features sampled at regular time intervals
        :param in_channels:      Number of input channels / features
        :param out_channels:     Number of output channels / features
        :param out_positions:    1D tensor with times (positions) at which to evaluate series
        :param d_embedding:      Latent dimensionality of the observation embeddings
        :param kernel_smoothing: Number of neighboring points used to estimate local inverse density
        """
        super(AsyncFeatureExtraction, self).__init__()
        self._in_channels = in_channels
        self._out_positions = out_positions.unsqueeze(0)
        self._max_position = torch.max(out_positions)
        self._kernel_smoothing = kernel_smoothing

        # encoding of input to embeddings
        self._dist_encoder = torch.nn.Linear(1, d_embedding)
        self._feat_encoder = torch.nn.Embedding(in_channels + 1, d_embedding)  # channel/value/time = 0 -> padding
        self._vals_encoder = torch.nn.Linear(1, d_embedding)

        # encodings to output
        self._linear = torch.nn.Linear(d_embedding * in_channels, out_channels)

        # as kernel scale is a free parameter we learn it along with the model
        self._kernel_scale = torch.nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _unpack(x):
        """ Unpacks input tensor """
        f = x[:, :, 0].long()
        v = x[:, :, 1]
        t = x[:, :, 2]
        return f, v, t

    def _encode_observations(self, x):
        """ Encodes (feature, value, time) tuples in x relative to each time point in t
        :param x:  Input tensor of shape (batch_size, num_observations, 3),
                   where each entry is a tuple of the form (feature, value, time)
        :returns:  Encodings of the observations in x relative to all timepoints in out_times
                   as a tensor with shape (batch_size, |out_times|, |x|, d_embedding)
        """
        f, v, t = self._unpack(x)

        # input 1: normalized relative distance to timepoints tin out_positions
        dists = (t.unsqueeze(1) - self._out_positions.unsqueeze(2)) / self._max_position
        d_encoded = self._dist_encoder(dists.unsqueeze(3))

        # input 2: feature / channel embedding
        f_encoded = self._feat_encoder(f).unsqueeze(1)

        # input 3: value embedding
        v_encoded = self._vals_encoder(v.unsqueeze(2)).unsqueeze(1)
        return d_encoded + f_encoded + v_encoded

    def _estimate_inv_density(self, x):
        """ Estimates for each observation x_i in x, a kernel-density estimate of the
        local density around x_i to discount clusters of close points.
        To minimize border effects, we use the minimum distance instead of the average
        For details, see wikipedia: https://en.wikipedia.org/wiki/Kernel_density_estimation
        :param x:  Input tensor of shape (batch_size, num_observations, 3),
                   where each entry is a tuple of the form (feature, value, time)
        :returns:  Density estimates of shape (batch_size, 1, |x|, 1)
        """
        f, _, t = self._unpack(x)

        # compute distances between all observations
        dists = torch.abs(t.unsqueeze(1) - t.unsqueeze(2))

        # mask out distance to oneself and observations of other channels
        # (we estimate the local density within each channel)
        within_channel = (f.unsqueeze(1) - f.unsqueeze(2)) == 0
        dists[~within_channel] = 1e10
        dists[dists == 0] = 1e10

        # Approximate density by average distance to nearest k neighbors
        # use min() shortcut if kernel_smoothing = 1
        if self._kernel_smoothing > 1:
            dists, _ = torch.sort(dists, dim=1, descending=False)
            inv_density = torch.mean(dists[:, :self._kernel_smoothing], dim=1, keepdim=True)
        else:
            inv_density, _ = torch.min(dists, dim=1, keepdim=True)

        return inv_density.unsqueeze(3).detach()

    def _causal_mask(self, x):
        """ Computes a causal mask to zero out future positions
        :returns: Boolean mask of shape (batch_size, |out_times|, |x|, 1)
        """
        mask = (x[:, :, 2].unsqueeze(1) - self._out_positions.unsqueeze(2)) <= 0
        return mask.unsqueeze(3).detach()

    @staticmethod
    def _padding_mask(x):
        """ Computes a padding mask to zero out padding positions
        :returns: Boolean mask of shape (batch_size, 1, |x|, 1)
        """
        return torch.any(x != 0, dim=2, keepdim=True).unsqueeze(1).detach()

    def forward(self, x):
        """ Forward pass through CKCNN
        :param x:  Input tensor of shape (batch_size, num_observations, 3),
                   where each entry is a tuple of the form (feature, value, time)
        :returns:  Tensor of shape (num_batches, |out_positions|, |out_channels|)
        """
        # encode each observation as a dense vector relative to each output position
        x_encoded = self._encode_observations(x.float())

        # create mask to drop padding and future observations (w.r.t to out positions)
        src_mask = torch.logical_and(self._causal_mask(x), self._padding_mask(x))

        # inverse local density (within-channel!)
        density = self._estimate_inv_density(x)

        encodings = []
        channels = x[:, :, 0].unsqueeze(1).unsqueeze(3)
        for f in range(self._in_channels):

            # mask out observations in the future or not in channel f
            channel_mask = (src_mask * (channels == f)).detach()

            # Normalize inverse density over number of observations in channel
            # accounting for future positions
            channel_density = (density * channel_mask) ** self._kernel_scale  # <- learn!
            channel_density = channel_density / (torch.sum(channel_density, dim=2, keepdim=True) + 1e-10)

            # for feature, weigh embeddings by their inverse density
            f_encoded = torch.sum(channel_density * x_encoded * channel_mask, dim=2) / (torch.sum(channel_mask, dim=2) + 1e-10)  # catch divide-by-zero
            encodings.append(f_encoded)

        # Reduce encoded feature set to out_channels
        y = torch.concat(encodings, dim=2).float()
        return self._linear(y).permute(0, 2, 1)  # channels -> dim=1


if __name__ == '__main__':
    # read in admission 200126 from MIMIC-III to use as toy example
    import pandas as pd
    df = pd.read_csv('../../../preprocessing/datasets/mimic-iii/non_aggregated_4h/mimic-iii_valid.csv')
    x = torch.tensor(df[df.episode == 200126][['x0', 'x1', 'x2']].values).unsqueeze(0).repeat(16, 1, 1)

    # evaluate at regular intervals from t=1h to t=72h
    t = torch.arange(1, 73).float()

    layer = AsyncFeatureExtraction(
        in_channels=46,
        out_channels=96,
        out_positions=t
    )
    print('out:', layer(x).shape)

    import matplotlib.pyplot as plt
    plt.imshow(layer(x).detach().numpy()[0])
    plt.show()