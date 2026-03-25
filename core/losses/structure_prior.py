import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-5
GCM = ((0.06, 0.63, 0.27), (0.3, 0.04, -0.35), (0.34, -0.6, 0.17))


def gaussian_basis_filters(scale, device, dtype, k=3):
    scale_tensor = torch.tensor(float(scale), device=device, dtype=dtype)
    std = torch.pow(torch.tensor(2.0, device=device, dtype=dtype), scale_tensor)
    filter_radius = int(torch.ceil(k * std + 0.5).item())
    coords = torch.arange(-filter_radius, filter_radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    g = torch.exp(-(xx / std) ** 2 / 2.0) * torch.exp(-(yy / std) ** 2 / 2.0)
    g = g / g.sum().clamp_min(EPS)

    dgdx = -(xx / (std**3 * 2.0 * math.pi)) * torch.exp(-(xx / std) ** 2 / 2.0) * torch.exp(-(yy / std) ** 2 / 2.0)
    dgdx = dgdx / dgdx.abs().sum().clamp_min(EPS)

    dgdy = -(yy / (std**3 * 2.0 * math.pi)) * torch.exp(-(yy / std) ** 2 / 2.0) * torch.exp(-(xx / std) ** 2 / 2.0)
    dgdy = dgdy / dgdy.abs().sum().clamp_min(EPS)

    return torch.stack([g, dgdx, dgdy], dim=0).unsqueeze(1)



def E_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    return Ex**2 + Ey**2 + Elx**2 + Ely**2 + Ellx**2 + Elly**2



def W_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Wx = Ex / (E + EPS)
    Wlx = Elx / (E + EPS)
    Wllx = Ellx / (E + EPS)
    Wy = Ey / (E + EPS)
    Wly = Ely / (E + EPS)
    Wlly = Elly / (E + EPS)
    return Wx**2 + Wy**2 + Wlx**2 + Wly**2 + Wllx**2 + Wlly**2



def C_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Clx = (Elx * E - El * Ex) / (E**2 + EPS)
    Cly = (Ely * E - El * Ey) / (E**2 + EPS)
    Cllx = (Ellx * E - Ell * Ex) / (E**2 + EPS)
    Clly = (Elly * E - Ell * Ey) / (E**2 + EPS)
    return Clx**2 + Cly**2 + Cllx**2 + Clly**2



def N_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Nlx = (Elx * E - El * Ex) / (E**2 + EPS)
    Nly = (Ely * E - El * Ey) / (E**2 + EPS)
    Nllx = (Ellx * E**2 - Ell * Ex * E - 2.0 * Elx * El * E + 2.0 * El**2 * Ex) / (E**3 + EPS)
    Nlly = (Elly * E**2 - Ell * Ey * E - 2.0 * Ely * El * E + 2.0 * El**2 * Ey) / (E**3 + EPS)
    return Nlx**2 + Nly**2 + Nllx**2 + Nlly**2



def H_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Hx = (Ell * Elx - El * Ellx) / (El**2 + Ell**2 + EPS)
    Hy = (Ell * Ely - El * Elly) / (El**2 + Ell**2 + EPS)
    return Hx**2 + Hy**2


INV_SWITCHER = {
    "E": E_inv,
    "W": W_inv,
    "C": C_inv,
    "N": N_inv,
    "H": H_inv,
}


class CIConv2d(nn.Module):
    def __init__(self, invariant="W", k=3, scale=0.8):
        super().__init__()
        if invariant not in INV_SWITCHER:
            raise ValueError(f"Unsupported invariant: {invariant}")
        self.invariant = invariant
        self.k = int(k)
        self.scale = float(scale)
        self.register_buffer("gcm", torch.tensor(GCM, dtype=torch.float32))

    def forward(self, batch):
        if batch.dim() != 4 or batch.shape[1] != 3:
            raise RuntimeError(f"CIConv2d expects BCHW RGB input, got {tuple(batch.shape)}")

        flat = batch.reshape(batch.shape[0], batch.shape[1], -1)
        transformed = torch.einsum("ij,bjk->bik", self.gcm.to(device=batch.device, dtype=batch.dtype), flat)
        transformed = transformed.reshape(batch.shape[0], 3, batch.shape[2], batch.shape[3])
        E, El, Ell = torch.split(transformed, 1, dim=1)

        basis = gaussian_basis_filters(self.scale, batch.device, batch.dtype, k=self.k)
        padding = basis.shape[-1] // 2
        E_out = F.conv2d(E, basis, padding=padding)
        El_out = F.conv2d(El, basis, padding=padding)
        Ell_out = F.conv2d(Ell, basis, padding=padding)

        E, Ex, Ey = torch.split(E_out, 1, dim=1)
        El, Elx, Ely = torch.split(El_out, 1, dim=1)
        Ell, Ellx, Elly = torch.split(Ell_out, 1, dim=1)

        invariant = INV_SWITCHER[self.invariant](E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly)
        return F.instance_norm(torch.log(invariant + EPS))



def build_structure_extractor(invariant="W", kernel_size=3, scale=0.8):
    extractor = CIConv2d(invariant=invariant, k=kernel_size, scale=scale)
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor
