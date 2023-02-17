import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def NT_Xent_aug(emb1, emb2, snr, batch_size, temperature=0.1, eps=1e-5):
    nor_emb1 = F.normalize(emb1, dim=1)
    nor_emb2 = F.normalize(emb2, dim=1)

    rep = torch.cat([nor_emb1, nor_emb2], dim=0)
    
    # (batch*2, batch*2, 3000)
    similarity_matrix = F.cosine_similarity(rep.unsqueeze(1), rep.unsqueeze(0), dim=-1).to('cpu') + eps
    similarity_matrix = torch.sum(similarity_matrix, dim=-1) / emb1.shape[1]

    snr_diff = torch.cat([snr, snr], dim=0).to('cpu')
    snr_diff = torch.log(torch.pow(2, torch.abs(torch.sub(snr_diff.unsqueeze(1), snr_diff.unsqueeze(0)))+eps))

    def l_ij(i, j):
        sim_ij = similarity_matrix[i, j].to(emb1.device)

        numerator = torch.exp(sim_ij / temperature)
        one_for_not_i = torch.ones((2 * batch_size, )).to(emb1.device).scatter_(0, torch.tensor([i]).to(emb1.device), 0.0)
        denominator = torch.sum(one_for_not_i * torch.exp(similarity_matrix[i, :].to(emb1.device) * snr_diff[i, :].to(emb1.device) / temperature))
      
        loss_ij = -torch.log(numerator / (denominator + eps))

        # print(f"numerator: {numerator}, denominator: {denominator}, loss: {loss_ij}")
        return loss_ij

    N = batch_size
    loss = 0.0
    for k in range(0, N):
        loss += l_ij(k, k+N) + l_ij(k+N, k)

    return 1.0 / (2*N) * loss

def NT_Xent_fn(emb, snr, batch_size, temperature=0.5, eps=1e-5, threshold=3):
    batch_size = emb.shape[0]
    nor_emb = F.normalize(emb, dim=1)
    
    # (batch*2, batch*2, 3000)
    similarity_matrix = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1) + eps
    similarity_matrix = torch.sum(similarity_matrix, dim=-1) / emb.shape[1]
    # print('sim_matrix: ', similarity_matrix)

    snr[torch.isnan(snr)] = -10
    snr[torch.isinf(snr)] = -10
    snr_diff_matrix = torch.abs(torch.sub(snr.unsqueeze(1), snr.unsqueeze(0)))
    pos = torch.where(snr_diff_matrix <= threshold, 1, 0)
    neg = torch.where(snr_diff_matrix > threshold, 1, 0)
    snr_diff = torch.log(torch.pow(2, snr_diff_matrix)+eps)
    # print('snr: ', snr)

    def l_ij(i):
        # print('snr: ', snr)
        # print('sim: ', similarity_matrix[i, :])
        # sim_ij = similarity_matrix[i, i].to(emb.device)
        # print('sim: ', sim_ij)
        numerator = torch.sum(pos[i, :] * torch.exp(similarity_matrix[i, :] * snr_diff[i, :].to(emb.device) / temperature))
        # numerator = torch.sum(pos[i, :] * torch.exp(similarity_matrix[i, :] / temperature))
        # print("numerator: ", numerator)
        denominator = torch.sum(neg[i, :] * torch.exp((similarity_matrix[i, :] * snr_diff[i, :].to(emb.device)) / temperature))
        # denominator = torch.sum(neg[i, :] * torch.exp((similarity_matrix[i, :]) / temperature))
        # print('denominator: ', denominator)
        loss = -torch.log(numerator / (denominator + eps))
        # print('loss: ', loss)
        return torch.sum(loss)
        
    loss = 0.0
    for k in range(0, batch_size):
        loss += l_ij(k)
            
    return 1.0 / batch_size * loss

def hierarchical_contrastive_loss(z1, z2=None, alpha=0.5, temporal_unit=0):
    if z2 is None:
        z2 = z1

    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def InfoNCE(query, pos, neg, temperature=0.1, negative_mode='paired'):
    # query: Auto-regressive model 產生出來目前的 context 'c'    (batch_size, emb_dim)
    # pos: 預測目標時間點的 representation (ground-truth)        (batch_size, emb_dim)
    # neg: 從其他 timesteps randomly sampled 的 representations  (num_negatives, emb_dim)
    query = query.to(neg.device)
    pos = pos.to(neg.device)

    pos_logits = torch.sum(query * pos, dim=1, keepdim=True)

    if negative_mode == 'unpaired':
        neg_logits = query @ neg.transpose(-2, -1)
    elif negative_mode == 'paired':
        query = query.unsqueeze(1)
        neg_logits = query @ neg.transpose(-2, -1)
        neg_logits = neg_logits.squeeze(1)

    logits = torch.cat([pos_logits, neg_logits], dim=1)

    labels = torch.zeros(len(logits), dtype=torch.long).to(neg.device)

    return F.cross_entropy(logits / temperature, labels)

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr.to(zjs.device)].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(zjs.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
