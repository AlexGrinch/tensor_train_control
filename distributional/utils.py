import torch
import torch.nn as nn


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def ce_with_logits(logits, target):
    return torch.sum(-target * torch.log_softmax(logits, -1), -1)


def categorical_loss(
    logits_t, logits_tp1, atoms_target_t, z, delta_z, v_min, v_max
):
    """
    Parameters
    ----------
    logits_t:        logits of categorical VD at (s_t, a_t)
    logits_tp1:      logits of categorical VD at (s_tp1, a_tp1)
    atoms_target_t:  target VD support
    z:               support of categorical VD at (s_t, a_t)
    delta_z:         fineness of categorical VD
    v_min, v_max:    left and right borders of catgorical VD
    """
    probs_tp1 = torch.softmax(logits_tp1, dim=-1)
    tz = torch.clamp(atoms_target_t, v_min, v_max)
    tz_z = torch.abs(tz[:, None, :] - z[None, :, None])
    tz_z = torch.clamp(1.0 - (tz_z / delta_z), 0., 1.)
    probs_target_t = torch.einsum("bij,bj->bi", (tz_z, probs_tp1)).detach()
    loss = ce_with_logits(logits_t, probs_target_t).mean()
    return loss


def quantile_loss(atoms_t, atoms_target_t, tau, num_atoms, criterion):
    """
    Parameters
    ----------
    atoms_t:         support of quantile VD at (s_t, a_t)
    atoms_target_t:  target VD support
    tau:             positions of quantiles where VD is approximated
    num_atoms:       number of atoms in quantile VD
    criterion:       loss function, usually Huber loss
    """
    atoms_diff = atoms_target_t[:, None, :] - atoms_t[:, :, None]
    delta_atoms_diff = atoms_diff.lt(0).to(torch.float32).detach()
    huber_weights = torch.abs(
        tau[None, :, None] - delta_atoms_diff
    ) / num_atoms
    loss = criterion(
        atoms_t[:, :, None], atoms_target_t[:, None, :], huber_weights
    ).mean()
    return loss


class HuberLoss(nn.Module):
    def __init__(self, clip_delta=1.0, reduction="elementwise_mean"):
        super(HuberLoss, self).__init__()
        self.clip_delta = clip_delta
        self.reduction = reduction or "none"

    def forward(self, y_pred, y_true, weights=None):
        td_error = y_true - y_pred
        td_error_abs = torch.abs(td_error)
        quadratic_part = torch.clamp(td_error_abs, max=self.clip_delta)
        linear_part = td_error_abs - quadratic_part
        loss = 0.5 * quadratic_part**2 + self.clip_delta * linear_part

        if weights is not None:
            loss = torch.mean(loss * weights, dim=1)
        else:
            loss = torch.mean(loss, dim=1)

        if self.reduction == "elementwise_mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss
