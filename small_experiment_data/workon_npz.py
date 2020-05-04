import numpy as np
import torch
import random

airplane_sdf=np.load('10ba42fc70f16d7f41d86c17c15247b0.npz')
positive= airplane_sdf['pos']
negative= airplane_sdf['neg']

#.npz files contain signed distance field, as a dictionary. 'pos' are the positive points (outside the shape), 'neg' are the negative points (inside the shape).
# positive and negative are build as follows: [x,y,z, distance]
print(np.shape(positive))
print(np.shape(negative))
print(positive[:,3])
print(negative[:,3])

#airplane= torch.Tensor(airplane_sdf)

def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data['pos']
    pos_tensor=torch.Tensor(pos_tensor)
    neg_tensor = data['neg']
    neg_tensor=torch.Tensor(neg_tensor)

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples




samples=unpack_sdf_samples_from_ram(airplane_sdf,4)
print(samples)