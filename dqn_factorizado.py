import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNFactorizado(nn.Module):
    def __init__(self, state_dim, emb_dim, n_skills, n_targets):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim)
        )
        
        # Head para skills
        self.skill_head = nn.Linear(emb_dim, n_skills)
        
        # Proyección por skill -> consulta para targets
        self.skill_query = nn.Linear(n_skills, emb_dim, bias=False)
        
        # Head para targets condicionada
        self.target_scorer = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # score de un target dado un skill
        )
        
        self.n_skills = n_skills
        self.n_targets = n_targets
        self.emb_dim = emb_dim

    def forward(self, state_vec, target_embs, action_mask=None):
        """
        state_vec: [B, state_dim]
        target_embs: [B, n_targets, emb_dim]  -> representación de héroes objetivo
        action_mask: [B, n_skills, n_targets] con 0/1
        """
        B = state_vec.size(0)
        
        # Representación global del estado
        h = self.state_encoder(state_vec)  # [B, emb_dim]
        
        # Q por skill
        q_skill = self.skill_head(h)  # [B, n_skills]
        
        # Construir consulta por skill
        skill_query = self.skill_query.weight.T  # [n_skills, emb_dim]
        skill_query = skill_query.unsqueeze(0).expand(B, -1, -1)  # [B, n_skills, emb_dim]
        
        # Expandir estado a cada target
        h_exp = h.unsqueeze(1).expand(B, self.n_targets, self.emb_dim)
        
        # Calcular Q_target para cada (skill, target)
        q_total = []
        for s in range(self.n_skills):
            q_s = q_skill[:, s].unsqueeze(1).expand(B, self.n_targets)  # [B, n_targets]
            q_s_query = skill_query[:, s, :].unsqueeze(1).expand(B, self.n_targets, self.emb_dim)
            cat = torch.cat([q_s_query, target_embs], dim=-1)  # [B, n_targets, 2*emb_dim]
            q_t = self.target_scorer(cat).squeeze(-1)  # [B, n_targets]
            q_total.append(q_s + q_t)
        
        q_total = torch.stack(q_total, dim=1)  # [B, n_skills, n_targets]
        
        # Aplicar máscara (0 = ilegal, 1 = legal)
        if action_mask is not None:
            illegal = (action_mask == 0)
            q_total = q_total.masked_fill(illegal, -1e9)
        
        return q_total  # matriz de Q-values por (skill, target)
