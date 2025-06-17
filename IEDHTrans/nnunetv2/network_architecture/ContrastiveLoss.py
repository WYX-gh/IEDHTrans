import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.eps = 1e-6

    def _dot_simililarity(self, x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, dim=1)
        return torch.matmul(x, y.T)

    def forward(self, features1, features2):
        device = features1.device
        batch_size = features1.size(0)

        if self.use_cosine_similarity:
            similarity_matrix = self._cosine_simililarity(features1, features2)
        else:
            similarity_matrix = self._dot_simililarity(features1, features2)

        labels = torch.arange(batch_size).to(device)
        logits = similarity_matrix / self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits[labels] - torch.log(exp_logits.sum(1) + self.eps)
        loss = -log_prob.mean()

        return loss

class HybridContrastiveLoss(nn.Module):
    def __init__(self, 
                 cross_modality_weight=0.5,
                 cross_scale_weight=0.5,
                 temperature=0.5,
                 projection_dim=256):
        super(HybridContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.weights = {
            'cross_modality': cross_modality_weight,
            'cross_scale': cross_scale_weight,
        }
        self.ntxent = NTXentLoss(temperature=temperature)

        # 支持 TokenSeg 的多尺度通道结构
        self.projection_layer_32 = nn.Linear(32, projection_dim)
        self.projection_layer_128= nn.Linear(128, projection_dim)
        self.projection_layer_512 = nn.Linear(512, projection_dim)

    def _contrastive_loss(self, f1, f2):
        B = f1.shape[0]
        f1_flat = f1.view(B, f1.shape[1], -1).mean(dim=2)
        f2_flat = f2.view(B, f2.shape[1], -1).mean(dim=2)

        # 支持 256 维通道
        if f1_flat.shape[1] == 32:
            f1_proj = self.projection_layer_32(f1_flat)
        elif f1_flat.shape[1] == 128:
            f1_proj = self.projection_layer_128(f1_flat)
        elif f1_flat.shape[1] == 512:
            f1_proj = self.projection_layer_512(f1_flat)
        else:
            raise ValueError(f"Unsupported channel dimension: {f1_flat.shape[1]}")

        if f2_flat.shape[1] == 32:
            f2_proj = self.projection_layer_32(f2_flat)
        elif f2_flat.shape[1] == 128:
            f2_proj = self.projection_layer_128(f2_flat)
        elif f2_flat.shape[1] == 512:
            f2_proj = self.projection_layer_512(f2_flat)
        else:
            raise ValueError(f"Unsupported channel dimension: {f2_flat.shape[1]}")

        return self.ntxent(f1_proj, f2_proj)

    def _compute_cross_modality_loss(self, features_list):
        loss = 0.0
        for f1, f2 in features_list:
            loss += self._contrastive_loss(f1, f2)
        return loss / len(features_list)

    def _compute_cross_scale_loss(self, features_list):
        loss = 0.0
        for f1, f2 in features_list:
            loss += self._contrastive_loss(f1, f2)
        return loss / len(features_list)

    def forward(self, features_dict):
        cross_modality_features = features_dict['cross_modality']
        cross_scale_features_mod1 = features_dict['cross_scale']['modality1']
        cross_scale_features_mod2 = features_dict['cross_scale']['modality2']

        cm_loss = self._compute_cross_modality_loss(cross_modality_features)
        cs_mod1_loss = self._compute_cross_scale_loss(cross_scale_features_mod1)
        cs_mod2_loss = self._compute_cross_scale_loss(cross_scale_features_mod2)

        loss = (
            self.weights['cross_modality'] * cm_loss +
            self.weights['cross_scale'] * (cs_mod1_loss + cs_mod2_loss) / 2
        )


        return loss