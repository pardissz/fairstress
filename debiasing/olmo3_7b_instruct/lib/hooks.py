import torch

class AddVectorHook:

    def __init__(self, vector: torch.Tensor, alpha: float):
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + self.alpha * self.vector.to(hs.dtype).to(hs.device)
            return (hs,) + output[1:]
        else:
            return output + self.alpha * self.vector.to(output.dtype).to(output.device)

class ProjectionHook:

    def __init__(self, apply_fn):
        self.apply_fn = apply_fn

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            orig_dtype = hs.dtype
            new_hs = self.apply_fn(hs.float()).to(orig_dtype)
            return (new_hs,) + output[1:]
        else:
            orig_dtype = output.dtype
            return self.apply_fn(output.float()).to(orig_dtype)

class ConditionalAddVectorHook:

    def __init__(self, condition_vec: torch.Tensor, threshold: float, behavior_vec: torch.Tensor, alpha: float):
        self.c = condition_vec
        self.threshold = threshold
        self.v = behavior_vec
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        orig_dtype = hs.dtype
        h = hs.float()
        c = self.c.to(dtype=torch.float32, device=h.device)
        v = self.v.to(dtype=torch.float32, device=h.device)
        c_sq_norm = (c * c).sum() + 1e-08
        proj = h @ c / c_sq_norm
        proj = proj.unsqueeze(-1) * c
        proj = torch.tanh(proj)
        h_norm = h.norm(dim=-1) + 1e-08
        proj_norm = proj.norm(dim=-1) + 1e-08
        sim = (h * proj).sum(-1) / (h_norm * proj_norm)
        gate = (sim > self.threshold).float().unsqueeze(-1)
        new_hs = (h + gate * self.alpha * v).to(orig_dtype)
        return (new_hs,) + output[1:] if is_tuple else new_hs

class InterventionModuleHook:

    def __init__(self, module):
        self.module = module

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if hs.shape[1] > 1:
            orig_dtype = hs.dtype
            last = hs[:, -1, :].float()
            new_last = self.module(last).to(orig_dtype)
            new_hs = hs.clone()
            new_hs[:, -1, :] = new_last
            return (new_hs,) + output[1:] if is_tuple else new_hs
        return output

class GatedInterventionModuleHook:

    def __init__(self, module, gate_w, gate_b, threshold=0.5):
        self.module = module
        self.gate_w = gate_w
        self.gate_b = gate_b
        self.threshold = threshold
        self.n_fired = 0
        self.n_total = 0

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if hs.shape[1] > 1:
            orig_dtype = hs.dtype
            last = hs[:, -1, :].float()
            gate_w = self.gate_w.to(device=last.device, dtype=torch.float32)
            gate_score = torch.sigmoid(last @ gate_w + self.gate_b)
            fire = gate_score > self.threshold
            self.n_total += fire.shape[0]
            self.n_fired += int(fire.sum().item())
            new_last = last.clone()
            if fire.any():
                edited = self.module(last[fire]).to(torch.float32)
                new_last[fire] = edited
            new_hs = hs.clone()
            new_hs[:, -1, :] = new_last.to(orig_dtype)
            return (new_hs,) + output[1:] if is_tuple else new_hs
        return output

def register_hook(model, layer_idx: int, hook_obj):
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_obj)
    return handle

def register_hooks_all_layers(model, hook_factory, layer_indices):
    handles = []
    for li in layer_indices:
        handles.append(register_hook(model, li, hook_factory(li)))
    return handles

def remove_hooks(handles):
    for h in handles:
        h.remove()
