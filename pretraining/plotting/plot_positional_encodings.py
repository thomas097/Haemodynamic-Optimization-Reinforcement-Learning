import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_pretrained(path):
    """ Load pretrained pytorch model from file
    :param path:  Path to Transformer instance in pt format
    :returns:     A PyTorch model
    """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.eval()
    return model


def get_positional_encoders(model):
    """ Crudely extracts positional encoding modules
    :param model: A Transformer instance
    :returns:     Dict mapping (block, head) to a RelativePositionalEncoding instance
    """
    pos_encoders = dict()

    # Extract positional encoders from each block and head
    for i, block in model.get_submodule('_encoder_layers').named_children():
        for j, self_attn_head in block.get_submodule('self_attn').get_submodule('_heads').named_children():
            pos_encoding = self_attn_head.get_submodule('_pos_encoding')
            pos_encoders[(int(i), int(j))] = self_attn_head.get_submodule('_pos_encoding')

    # Extract positional encoders from fusion layer
    try:
        for j, self_attn_head in model.get_submodule('_fusion').get_submodule('self_attn').get_submodule('_heads').named_children():
            pos_encoding = self_attn_head.get_submodule('_pos_encoding')
            pos_encoders[(int(i) + 1, int(j))] = self_attn_head.get_submodule('_pos_encoding')
    except:
        pass

    return pos_encoders


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def visualize_rel_positions(pos_encoders, n_blocks, n_heads, max_dist=72, lookback=72, use_softmax=False, steps=1000):
    """ Visualizes the relative positions matrix by feeding model a sequence
        of relative positions and observing the encoder's response curve
    :param pos_encoders:  Dict mapping (block, head) to a RelativePositionalEncoding instance
    :param n_blocks:      Number of blocks in Transformer
    :param n_heads:       Number of heads in Transformer
    :param max_dist:      Maximum distance to look into the past from current observation (i.e. now - observation_time)
    :param use_softmax:   Whether to incorporate the softmax operation as in the attention mechanism
    :param steps:         Number of points along range (set to high value)
    """
    # define rel pos vector with relative (absolute) distances in specified range
    rel_pos = torch.linspace(-1, 0, steps).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    plt.figure(figsize=(12, 8))
    for block in range(n_blocks):
        for head in range(n_heads):
            pos_encoder = pos_encoders[(block, head)](rel_pos)

            # observe response of positional encoder
            response = -torch.squeeze(pos_encoder).detach().numpy() # pos encoding is subtracted from attention matrix!
            times = torch.squeeze(rel_pos).detach().numpy()         # bigger relative distance -> further in the past

            if use_softmax:
                response = softmax(response)

            # look back only x hours
            i = int(steps * lookback / max_dist)

            # Plot response as a function of time
            plt.subplot(n_blocks, n_heads, n_heads * block + head + 1)
            plt.plot(times[-i:], response[-i:])
            plt.title('block: %d - head: %d' % (block, head))
            plt.xlabel('Time (h)')
            plt.ylabel('Response')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model = load_pretrained("../results/transformer_nsp_pretraining_00001/encoder.pt")
    pos_encoders = get_positional_encoders(model)

    visualize_rel_positions(pos_encoders, n_blocks=2, n_heads=4, max_dist=72, steps=1000, lookback=12, use_softmax=True)