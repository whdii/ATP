import torch
import torch.nn as nn


class ActivationTracker:
    def __init__(self, model, model_bac, model_name):
        self.model = model
        self.model_bac = model_bac
        self.model_name = model_name
        self.features_handles = []
        self.loss_features = {'main': [], 'l1': [], 'bac': [], 'l1_bac': []}
        self.loss_gradients = {'main': [], 'l1': [], 'bac': [], 'l1_bac': []}
        self.activations = []
        self.statistics = {
            'results_std': None, 'magnitude_std': None,
            'results_std_l1': None, 'magnitude_std_l1': None,
            'results_std_bac': None, 'magnitude_std_bac': None,
            'results_std_l1_bac': None, 'magnitude_std_l1_bac': None
        }

        self._register_hooks()

    def _register_hooks(self):
        layers = [(self.model, 'main'), (self.model_bac, 'bac')]
        for model, key in layers:
            self.features_handles.append(model.layer4.register_forward_hook(self._create_forward_hook(key)))
            self.features_handles.append(model.layer4.register_backward_hook(self._create_backward_hook(key)))
            self.features_handles.append(model.layer1.register_forward_hook(self._create_forward_hook(f'l1_{key}')))
            self.features_handles.append(model.layer1.register_backward_hook(self._create_backward_hook(f'l1_{key}')))

    def _create_forward_hook(self, key):
        def hook(module, input, output):
            self.loss_features[key].append(output)
        return hook

    def _create_backward_hook(self, key):
        def hook(module, grad_in, grad_out):
            self.loss_gradients[key].append(grad_out[0])
        return hook

    def clear_hooks(self):
        for handle in self.features_handles:
            handle.remove()
        self.features_handles = []

    def calculate_statistics(self, feat_out, batch_idx, statis_results_std, magnitude_std):
        if len(feat_out.shape) == 4:
            N, C, H, W = feat_out.shape
            feat_out = feat_out.view(N, C, H * W).mean(dim=-1)

        count_activate = (feat_out > 1e-2 * feat_out.max(dim=1, keepdim=True)[0]).sum(dim=0).cpu().numpy()
        feat_mean_magnitude = (feat_out.sum(dim=0) / count_activate).cpu().numpy()

        if batch_idx == 0:
            statis_results_std, magnitude_std = count_activate, feat_mean_magnitude
        else:
            statis_results_std += count_activate
            magnitude_std = (magnitude_std + feat_mean_magnitude) / 2

        return statis_results_std, magnitude_std

    def get_statistics(self, data, batch_idx):
        self.model(data)
        self.model_bac(data)

        for key in self.loss_features:
            feat_out = self.loss_features[key][0]
            stats_key = 'results_' + key.replace('bac', 'std_bac')
            mag_key = 'magnitude_' + key.replace('bac', 'std_bac')
            self.statistics[stats_key], self.statistics[mag_key] = self.calculate_statistics(
                feat_out, batch_idx,
                self.statistics[stats_key] if batch_idx > 0 else None,
                self.statistics[mag_key] if batch_idx > 0 else None
            )

        return self.statistics

    def run(self, latent_r_bac, robust_index):
        robust_feat_out = latent_r_bac[:, robust_index, ...]
        robust_feat_out = robust_feat_out.view(*robust_feat_out.shape[:2], -1).mean(dim=-1)
        robust_large_feat = torch.sigmoid(100 * (robust_feat_out - 1e-2 * robust_feat_out.max(dim=1, keepdim=True)[0])).sum(dim=0).float()
        return robust_large_feat.mean()

    def loss_f(self, outputs, labels):
        device = outputs[0].device
        one_hot_labels = torch.eye(len(outputs[0]), device=device)[labels].to(device)
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        return torch.clamp(-1 * (i - j), min=0)

    def loss_function(self, input, target, robust_idx, non_robust_idx, robust_idx_l1, non_robust_idx_l1):
        self.model.zero_grad()
        self.loss_features = {key: [] for key in self.loss_features}
        self.loss_gradients = {key: [] for key in self.loss_gradients}

        output = self.model(input)
        target = output.max(1)[1]
        loss_ff = self.loss_f(output, target).sum()

        latent_r_adv = self.loss_features['l1'][0].requires_grad_(True)
        latent_r_bac = self.loss_features['main'][0].requires_grad_(True)
        latent_r_bac_l1 = self.loss_features['l1_bac'][0].requires_grad_(True)

        loss_con = self.run(latent_r_bac, robust_idx)
        grad_latent_NR = torch.autograd.grad(-loss_ff, latent_r_adv, retain_graph=True, create_graph=False)[0]
        grad_latent_R_l1 = torch.autograd.grad(-loss_con, latent_r_bac_l1, retain_graph=True, create_graph=False)[0]

        robust_feature_grad_l1 = latent_r_bac_l1.index_select(1, torch.tensor(robust_idx_l1).to(input.device)) * grad_latent_R_l1.index_select(1, torch.tensor(robust_idx_l1).to(input.device))
        loss_bac = -torch.log(torch.sum(torch.abs(robust_feature_grad_l1)))

        non_robust_feature_grad = latent_r_adv.index_select(1, torch.tensor(non_robust_idx_l1).to(input.device)) * grad_latent_NR.index_select(1, torch.tensor(non_robust_idx_l1).to(input.device))
        loss_adv = -torch.log(torch.sum(torch.abs(non_robust_feature_grad)))

        loss = loss_adv + loss_bac - torch.log(loss_con) + torch.log(loss_ff)
        return loss, output, loss_ff
