from dataclasses import dataclass
from typing import Tuple, Optional, Deque
from collections import deque
import random
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
#  Configuración
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM = 256     # <- ajusta al tamaño real de tu vector de estado
EMB_DIM = 64        # dimensión del embedding interno del modelo
N_SKILLS = 6        # p.ej.: BASIC + 4 specials + MASTER
N_TARGETS = 6       # p.ej.: 3v3 => hasta 6 posibles objetivos

LR = 3e-4
GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_SIZE = 100_000
LEARN_START = 2_000
TARGET_UPDATE_RATE = 0.01   # soft update (τ)
MAX_GRAD_NORM = 5.0

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 150_000

TRAINING_STEPS = 300_000
EVAL_EVERY = 5_000
SAVE_EVERY = 25_000

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
#  Red: DQN factorizado (Q_skill + Q_target condicionado)
# ============================================================

class DQNFactorizado(nn.Module):
    def __init__(self, state_dim: int, emb_dim: int, n_skills: int, n_targets: int):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim),
            nn.ReLU(),
        )
        # Head para skills (Q_skill)
        self.skill_head = nn.Linear(emb_dim, n_skills)
        # Matriz de consultas por skill (cada skill tiene un vector de consulta)
        self.skill_query = nn.Linear(n_skills, emb_dim, bias=False)
        # Scorer para targets condicionado por skill
        self.target_scorer = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.n_skills = n_skills
        self.n_targets = n_targets
        self.emb_dim = emb_dim

    def forward(self, state_vec: torch.Tensor, target_embs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        state_vec: [B, state_dim]
        target_embs: [B, n_targets, emb_dim]
        action_mask: [B, n_skills, n_targets]  (0/1). Si se provee, se enmascara con -1e9.
        return: Q_total [B, n_skills, n_targets]
        """
        B = state_vec.size(0)
        h = self.state_encoder(state_vec)  # [B, emb_dim]
        q_skill = self.skill_head(h)       # [B, n_skills]

        # Construir consulta por skill (parámetro fijo de la red)
        skill_query = self.skill_query.weight.T              # [n_skills, emb_dim]
        skill_query = skill_query.unsqueeze(0).expand(B, -1, -1)  # [B, n_skills, emb_dim]

        q_total_list = []
        for s in range(self.n_skills):
            # Q base de la skill s
            q_s = q_skill[:, s].unsqueeze(1).expand(B, self.n_targets)  # [B, n_targets]
            # Consulta específica de la skill s
            q_s_query = skill_query[:, s, :].unsqueeze(1).expand(B, self.n_targets, self.emb_dim)
            # Concatenar consulta de skill con embeddings de target
            cat = torch.cat([q_s_query, target_embs], dim=-1)           # [B, n_targets, 2*emb_dim]
            q_t = self.target_scorer(cat).squeeze(-1)                   # [B, n_targets]
            q_total_list.append(q_s + q_t)

        q_total = torch.stack(q_total_list, dim=1)  # [B, n_skills, n_targets]

        if action_mask is not None:
            illegal = (action_mask == 0)
            q_total = q_total.masked_fill(illegal, -1e9)
        return q_total

# ============================================================
#  Utilidades
# ============================================================

def flatten_action(skill_idx: torch.Tensor, target_idx: torch.Tensor, n_targets: int) -> torch.Tensor:
    """Convierte (skill, target) en índice aplanado."""
    return skill_idx * n_targets + target_idx

def unflatten_action(flat_idx: torch.Tensor, n_targets: int) -> Tuple[torch.Tensor, torch.Tensor]:
    skill_idx = flat_idx // n_targets
    target_idx = flat_idx % n_targets
    return skill_idx, target_idx


def select_action(q_values: torch.Tensor, action_mask: torch.Tensor, epsilon: float) -> Tuple[int, int, int]:
    """
    q_values: [1, n_skills, n_targets]
    action_mask: [1, n_skills, n_targets]
    Devuelve: (flat_idx, skill_idx, target_idx)
    """
    q = q_values[0]
    mask = action_mask[0].bool()
    legal_flat = mask.view(-1)
    if legal_flat.sum() == 0:
        # No hay acciones legales; devolvemos algo por defecto
        return 0, 0, 0
    if random.random() < epsilon:
        # Explora entre legales
        legal_indices = torch.nonzero(legal_flat, as_tuple=False).squeeze(1)
        flat_idx = legal_indices[torch.randint(len(legal_indices), (1,))].item()
    else:
        # Explotación
        q_flat = q.view(-1)
        # Poner -inf donde sea ilegal
        q_flat_masked = q_flat.clone()
        q_flat_masked[~legal_flat] = -1e9
        flat_idx = int(torch.argmax(q_flat_masked).item())
    skill_idx, target_idx = unflatten_action(torch.tensor(flat_idx), N_TARGETS)
    return flat_idx, int(skill_idx.item()), int(target_idx.item())


# ============================================================
#  Replay Buffer simple (1-step)
# ============================================================

@dataclass
class Transition:
    state_vec: torch.Tensor          # [state_dim]
    target_embs: torch.Tensor        # [n_targets, emb_dim]
    action_flat: int                 # skill* n_targets + target
    action_mask: torch.Tensor        # [n_skills, n_targets]
    reward: float
    next_state_vec: torch.Tensor     # [state_dim]
    next_target_embs: torch.Tensor   # [n_targets, emb_dim]
    next_action_mask: torch.Tensor   # [n_skills, n_targets]
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        # Empaquetar a tensores
        state_vec = torch.stack([b.state_vec for b in batch], dim=0).to(device)
        target_embs = torch.stack([b.target_embs for b in batch], dim=0).to(device)
        action_flat = torch.tensor([b.action_flat for b in batch], dtype=torch.long, device=device)
        action_mask = torch.stack([b.action_mask for b in batch], dim=0).to(device)
        reward = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=device)
        next_state_vec = torch.stack([b.next_state_vec for b in batch], dim=0).to(device)
        next_target_embs = torch.stack([b.next_target_embs for b in batch], dim=0).to(device)
        next_action_mask = torch.stack([b.next_action_mask for b in batch], dim=0).to(device)
        done = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device)
        return (state_vec, target_embs, action_flat, action_mask,
                reward, next_state_vec, next_target_embs, next_action_mask, done)

    def __len__(self):
        return len(self.buffer)

# ============================================================
#  Agente DQN
# ============================================================

class DQNAgent:
    def __init__(self, state_dim: int, emb_dim: int, n_skills: int, n_targets: int, lr: float = 3e-4):
        self.online = DQNFactorizado(state_dim, emb_dim, n_skills, n_targets).to(device)
        self.target = DQNFactorizado(state_dim, emb_dim, n_skills, n_targets).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.n_targets = n_targets
        self.loss_fn = nn.SmoothL1Loss()

    @torch.no_grad()
    def q_values(self, state_vec: torch.Tensor, target_embs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        return self.online(state_vec, target_embs, action_mask)

    def train_step(self, batch, gamma: float = 0.99):
        (state_vec, target_embs, action_flat, action_mask,
         reward, next_state_vec, next_target_embs, next_action_mask, done) = batch

        # Q(s,a) actual
        q = self.online(state_vec, target_embs, action_mask)         # [B, S, T]
        q_flat = q.view(q.size(0), -1)                               # [B, S*T]
        q_sa = q_flat.gather(1, action_flat.unsqueeze(1)).squeeze(1) # [B]

        with torch.no_grad():
            # Double DQN: argmax con online, eval con target
            q_next_online = self.online(next_state_vec, next_target_embs, next_action_mask)  # [B,S,T]
            q_next_online_flat = q_next_online.view(q_next_online.size(0), -1)
            # enmascarado ya aplicado en forward; por seguridad:
            legal_next = next_action_mask.view(next_action_mask.size(0), -1).bool()
            q_next_online_flat_masked = q_next_online_flat.clone()
            q_next_online_flat_masked[~legal_next] = -1e9
            a_prime = torch.argmax(q_next_online_flat_masked, dim=1)  # [B]

            q_next_target = self.target(next_state_vec, next_target_embs, next_action_mask)  # [B,S,T]
            q_next_target_flat = q_next_target.view(q_next_target.size(0), -1)
            max_q_next = q_next_target_flat.gather(1, a_prime.unsqueeze(1)).squeeze(1)

            target = reward + gamma * (1.0 - done) * max_q_next

        loss = self.loss_fn(q_sa, target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), MAX_GRAD_NORM)
        self.optim.step()
        return float(loss.item()), float(q_sa.mean().item()), float(target.mean().item())

    def soft_update(self, tau: float = 0.01):
        with torch.no_grad():
            for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p_o.data)

    def save(self, path: str):
        torch.save({
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'optim': self.optim.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=device)
        self.online.load_state_dict(ckpt['online'])
        self.target.load_state_dict(ckpt['target'])
        self.optim.load_state_dict(ckpt['optim'])

# ============================================================
#  Entorno de ejemplo (sustituye por tu integración real)
# ============================================================

class FakeEnv:
    """Entorno simulado para probar la plantilla.
    Reemplaza por tu motor real: observa tu MatchState y construye
    state_vec, target_embs, action_mask, y aplica la acción en step().
    """
    def __init__(self, state_dim: int, n_targets: int, emb_dim: int):
        self.state_dim = state_dim
        self.n_targets = n_targets
        self.emb_dim = emb_dim
        self.t = 0

    def reset(self):
        self.t = 0
        state_vec = torch.randn(self.state_dim)
        target_embs = torch.randn(self.n_targets, self.emb_dim)
        # Genera una máscara con ~70% legales
        mask = (torch.rand(N_SKILLS, self.n_targets) > 0.3).float()
        return state_vec, target_embs, mask

    def step(self, action_flat: int):
        self.t += 1
        # Dinámica ficticia
        reward = random.uniform(-0.1, 0.2)
        done = (self.t >= 50) or (random.random() < 0.02)
        next_state_vec = torch.randn(self.state_dim)
        next_target_embs = torch.randn(self.n_targets, self.emb_dim)
        next_mask = (torch.rand(N_SKILLS, self.n_targets) > 0.3).float()
        return next_state_vec, next_target_embs, next_mask, reward, done, {}

# ============================================================
#  Bucle de entrenamiento
# ============================================================

def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return eps_end
    frac = step / float(decay_steps)
    return eps_start + (eps_end - eps_start) * frac


def train_loop():
    env = FakeEnv(STATE_DIM, N_TARGETS, EMB_DIM)
    agent = DQNAgent(STATE_DIM, EMB_DIM, N_SKILLS, N_TARGETS, lr=LR)
    rb = ReplayBuffer(REPLAY_SIZE)

    state_vec, target_embs, action_mask = env.reset()
    global_step = 0
    episode = 0

    while global_step < TRAINING_STEPS:
        agent.online.train()
        epsilon = linear_epsilon(global_step, EPS_START, EPS_END, EPS_DECAY_STEPS)

        # Formatear a batch=1
        s = state_vec.unsqueeze(0).to(device)
        te = target_embs.unsqueeze(0).to(device)
        am = action_mask.unsqueeze(0).to(device)

        with torch.no_grad():
            q_vals = agent.q_values(s, te, am)
        a_flat, a_skill, a_target = select_action(q_vals, am, epsilon)

        next_state_vec, next_target_embs, next_action_mask, reward, done, _ = env.step(a_flat)

        rb.push(state_vec, target_embs, a_flat, action_mask, reward,
                next_state_vec, next_target_embs, next_action_mask, done)

        state_vec, target_embs, action_mask = next_state_vec, next_target_embs, next_action_mask

        # Entrenamiento
        if len(rb) >= max(BATCH_SIZE, LEARN_START):
            batch = rb.sample(BATCH_SIZE)
            loss, q_mean, target_mean = agent.train_step(batch, GAMMA)
            agent.soft_update(TARGET_UPDATE_RATE)
        else:
            loss, q_mean, target_mean = float('nan'), float('nan'), float('nan')

        # Logging simple
        if (global_step + 1) % 1000 == 0:
            print(f"step={global_step+1} eps={epsilon:.3f} loss={loss:.4f} q={q_mean:.2f} tgt={target_mean:.2f} rb={len(rb)}")

        # Eval ficticia
        if (global_step + 1) % EVAL_EVERY == 0:
            eval_return = evaluate(agent, episodes=5)
            print(f"[EVAL] step={global_step+1} return={eval_return:.3f}")

        if (global_step + 1) % SAVE_EVERY == 0:
            path = f"dqn_factorizado_step_{global_step+1}.pt"
            agent.save(path)
            print(f"[CKPT] guardado en {path}")

        if done:
            episode += 1
            state_vec, target_embs, action_mask = env.reset()

        global_step += 1

    # Guardado final
    agent.save("dqn_factorizado_final.pt")
    print("Entrenamiento finalizado y modelo guardado.")


@torch.no_grad()
def evaluate(agent: DQNAgent, episodes: int = 5) -> float:
    agent.online.eval()
    env = FakeEnv(STATE_DIM, N_TARGETS, EMB_DIM)
    total_return = 0.0
    for _ in range(episodes):
        state_vec, target_embs, action_mask = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            s = state_vec.unsqueeze(0).to(device)
            te = target_embs.unsqueeze(0).to(device)
            am = action_mask.unsqueeze(0).to(device)
            q_vals = agent.q_values(s, te, am)
            # Greedy puro en evaluación (epsilon=0)
            a_flat, _, _ = select_action(q_vals, am, epsilon=0.0)
            state_vec, target_embs, action_mask, reward, done, _ = env.step(a_flat)
            episode_return += reward
        total_return += episode_return
    return total_return / episodes


if __name__ == "__main__":
    train_loop()
