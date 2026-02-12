import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, R, alpha):
        super().__init__()

        self.d_in = d_in
        self.d_hid = R * d_in
        self.alpha = alpha

        self.M = nn.Parameter(torch.empty(self.d_hid, self.d_in))
        self.b = nn.Parameter(torch.zeros(self.d_hid))

        nn.init.uniform_(self.M, -0.1, 0.1)

        self.normalize_rows()

    def forward(self, x):
        c = F.relu(x @ self.M.t() + self.b)
        x_hat = c @ self.M
        return x_hat, c

    def loss(self, x, x_hat, c):
        reconstruction_loss = (x_hat - x).pow(2).sum(dim=1).mean()
        sparse_loss = self.alpha * c.abs().sum(dim=1).mean()
        return reconstruction_loss + sparse_loss

    @torch.no_grad()
    def normalize_rows(self):
        norms = self.M.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.M.div_(norms)

def save_path(sae, epochs, base_dir="models"):
    os.makedirs(base_dir, exist_ok=True)
    model = "sae"

    while True:
        name = f"{model}_din{sae.d_in}_dhid{sae.d_hid}_e{epochs}_alpha{sae.alpha}.pt"
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            return path
        run += 1
        

def train(sae, x_full, lr=1e-3, epochs=3, batch_size=1024):
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    N = x_full.size(0)
    step = 0

    for epoch in range(epochs):
        perm = torch.randperm(N, device=x_full.device)

        for start in range(0, N, batch_size):
            step += 1
            idx = perm[start:start + batch_size]
            x_batch = x_full[idx]

            x_hat, c = sae(x_batch)
            loss = sae.loss(x_batch, x_hat, c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sae.normalize_rows()

            if step % 200 == 0:
                print(f"epoch {epoch}, step {step}, loss {loss.item():.6f}")
    
    torch.save(sae.state_dict(), save_path(sae, epochs))
