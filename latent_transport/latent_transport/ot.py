import torch


def pairwise_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2) ** 2


def sinkhorn_coupling(x: torch.Tensor, y: torch.Tensor, eps: float = 0.2, iters: int = 20) -> torch.Tensor:
    B, N, _ = x.shape
    M = y.shape[1]
    C = pairwise_cost(x, y)
    K = torch.exp(-C / eps).clamp_min(1e-9)
    a = torch.full((B, N), 1.0 / N, device=x.device, dtype=x.dtype)
    b = torch.full((B, M), 1.0 / M, device=x.device, dtype=x.dtype)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iters):
        Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        u = a / Kv
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        v = b / KTu
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)
    return P / P.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)


def barycentric_targets(P: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    row_sum = P.sum(dim=2, keepdim=True).clamp_min(1e-8)
    return torch.bmm(P / row_sum, y)


def sample_ot_targets(P: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Sample one target point from each OT row rather than taking a row barycenter.
    This avoids mode-averaging in multimodal settings.
    P: [B, N, M]
    y: [B, M, D]
    returns: [B, N, D]
    """
    B, N, M = P.shape
    D = y.shape[-1]
    row = P / P.sum(dim=2, keepdim=True).clamp_min(1e-8)
    idx = torch.multinomial(row.reshape(B * N, M), 1).reshape(B, N)
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(y, 1, gather_idx)


def sinkhorn_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 0.2, iters: int = 20) -> torch.Tensor:
    P = sinkhorn_coupling(x, y, eps=eps, iters=iters)
    C = pairwise_cost(x, y)
    return (P * C).sum(dim=(1, 2)).mean()