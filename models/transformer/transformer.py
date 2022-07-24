from matplotlib import image
import torch
import torch.nn.functional as F
from torch import nn
import copy
from models.containers import ModuleList
from models.transformer.utils import sinusoid_encoding_table
from models.beam_search import *
from ..captioning_model import CaptioningModel


class SP(nn.Module):
    """SP layer implementation
    
    Args:
        num_clusters : int
            The number of pseudo regions
        dim : int
            Dimension of pseudo regions
        alpha : float
            Parameter of initialization. Larger value is harder assignment.
        normalize_input : bool
            If true, pseudo regions-wise L2 normalization is applied to input.
    """
    def __init__(self, num_regions=64, dim=128, alpha=100.0, normalize_input=True):
        super().__init__()
        self.num_regions = num_regions
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_regions, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_regions, dim))
        self.init_weights()
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, grids):
                   
        N, C = grids.shape[0], grids.shape[-1]

        grids = grids.view(N, 7, 7, -1).permute(0,3,1,2).contiguous()

        if self.normalize_input:
            grids = F.normalize(grids, p=2, dim=1)  # across descriptor dim

        soft_assign = self.conv(grids).view(N, self.num_regions, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = grids.view(N, C, -1)
        
        residual = x_flatten.expand(self.num_regions, -1, -1, -1).permute(1, 0, 2, 3).contiguous() - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).contiguous().unsqueeze(0)

        residual *= soft_assign.unsqueeze(2)
        p = residual.sum(dim=-1)

        p = F.normalize(p, p=2, dim=2)  # intra-normalization
        p = p.view(grids.size(0), -1)
        p = F.normalize(p, p=2, dim=1)  # L2 normalize

        return p

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder,num_clusters, vocab_size, max_len, padding_idx, text_d_model=512):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.text_d_model = text_d_model
        self.num_clusters=num_clusters
        self.padding_idx = padding_idx
        self.word_emb = nn.Embedding(vocab_size, text_d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, text_d_model, 0), freeze=True)

        self.SP = SP(num_regions=self.num_clusters, dim=2048)

        self.softmax = nn.Softmax(dim=-1)
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mode, images, seq=None, max_len=None, eos_idx=None, beam_size=None, out_size=1, return_probs=False):
        '''
        images: torch.Size([50, 49, 2048])
        seq: torch.Size([50, 27])
        '''
        if mode == 'xe':
            bs, _, vis_dim = images.size()
            # Grid feature
            grid_enc_output, grid_mask_enc = self.encoder(images)

            # Pseudo-region feature
            pseudo_region = self.SP(images).view(bs, -1, vis_dim) # (N, num_clusters*2048) -> (N, num_clusters, 2048)
            pseudo_region_enc_output, pseudo_region_mask_enc = self.encoder(pseudo_region)

            output, mask = torch.cat([grid_enc_output, pseudo_region_enc_output],dim=1), torch.cat([grid_mask_enc, pseudo_region_mask_enc], dim=-1)
            dec_output = self.decoder(seq, output, mask)

            return dec_output

        elif mode == 'rl':
            bs = BeamSearch(self, max_len, eos_idx, beam_size)
            return bs.apply(images, out_size, return_probs)
        
    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]
    
    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                grid_enc_output, grid_mask_enc = self.encoder(visual)
                bs, _, vis_dim = visual.size()
                pseudo_region = self.SP(visual).view(bs, -1, vis_dim)
                pseudo_region_enc_output, pseudo_region_mask_enc = self.encoder(pseudo_region)
                self.enc_output, self.mask_enc = torch.cat([grid_enc_output, pseudo_region_enc_output],dim=1), torch.cat([grid_mask_enc, pseudo_region_mask_enc], dim=-1)
                
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long() # self.bos_idx: '<bos>'
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output
        return self.decoder(it, self.enc_output, self.mask_enc)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
