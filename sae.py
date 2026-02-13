import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, R: int, alpha: float):
        super().__init__()

        self.d_in = d_in
        self.R = R
        self.d_hid = R * d_in
        self.alpha = alpha

        self.M = nn.Parameter(torch.empty(self.d_hid, self.d_in))
        self.b = nn.Parameter(torch.zeros(self.d_hid))

        nn.init.uniform_(self.M, -0.1, 0.1)

        self.normalize_rows()

    def forward(self, x: torch.Tensor):
        c = F.relu(x @ self.M.t() + self.b)
        x_hat = c @ self.M
        return x_hat, c

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, c: torch.Tensor):
        reconstruction_loss = (x_hat - x).pow(2).sum(dim=1).mean()
        sparse_loss = self.alpha * c.abs().sum(dim=1).mean()
        return reconstruction_loss + sparse_loss

    @torch.no_grad()
    def normalize_rows(self):
        norms = self.M.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.M.div_(norms)


def load_activations(acts_path: str, device: torch.device, dtype: torch.dtype = torch.float32):
    x = torch.load(acts_path, map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected tensor in {acts_path}, got {type(x)}")
    if x.dim() != 2:
        raise ValueError(f"Expected [N, D] tensor in {acts_path}, got {tuple(x.shape)}")
    return x.to(device=device, dtype=dtype)


def save_path(sae: SparseAutoencoder, epochs: int, base_dir: str = "saes"):
    os.makedirs(base_dir, exist_ok=True)
    run = 0
    while True:
        name = f"sae_din{sae.d_in}_R{sae.R}_alpha{sae.alpha}_e{epochs}_r{run}.pt"
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            return path
        run += 1
        

def train(
    sae: SparseAutoencoder,
    x_full: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 3,
    batch_size: int = 1024,
    save_dir: str = "saes",
    log_every: int = 200,
):
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    N = x_full.size(0)
    step = 0

    sae.train()

    for epoch in range(epochs):
        perm = torch.randperm(N, device=x_full.device)

        for start in range(0, N, batch_size):
            step += 1
            idx = perm[start:start + batch_size]
            x_batch = x_full[idx]

            x_hat, c = sae(x_batch)
            loss = sae.loss(x_batch, x_hat, c)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sae.normalize_rows()

            if log_every and (step % log_every == 0):
                print(f"epoch {epoch}, step {step}, loss {loss.item():.6f}")

    out_path = save_path(sae, epochs, base_dir=save_dir)
    torch.save(sae.state_dict(), out_path)
    print("Saved SAE to:", out_path)
    return out_path


if __name__ == "__main__":
    acts_path = "activations/gcn_l3_h256_r0/layer_0_postrelu.pt"

    R = 2
    alpha = 1e-2
    lr = 1e-3
    epochs = 100
    batch_size = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    x_full = load_activations(acts_path, device=device, dtype=torch.float32)
    d_in = x_full.size(1)
    print("Loaded activations:", tuple(x_full.shape), "d_in =", d_in)

    sae = SparseAutoencoder(d_in=d_in, R=R, alpha=alpha).to(device)

    train(sae, x_full, lr=lr, epochs=epochs, batch_size=batch_size)
