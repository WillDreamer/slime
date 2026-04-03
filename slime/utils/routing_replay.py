import logging
import os

import torch

logger = logging.getLogger(__name__)

ROUTING_REPLAY = None


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()


class TrainingExpertBalanceTracker:
    """Collects expert routing statistics during Megatron training forward passes.

    Call ``record(layer_idx, top_indices, num_experts)`` during each forward pass.
    After the step, call ``get_metrics_and_reset()`` to retrieve per-layer and
    summary balance metrics for logging (e.g. to wandb).
    """

    _instance = None

    def __init__(self):
        # layer_idx -> list of top_indices tensors accumulated over microbatches
        self._layer_data: dict[int, list[torch.Tensor]] = {}
        self._num_experts: int | None = None
        self._enabled = os.environ.get("MOE_BALANCE_TRACKING", "0") == "1"
        self._layer_counter = 0  # auto-increment layer index within a step

    @classmethod
    def get_instance(cls) -> "TrainingExpertBalanceTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(self, top_indices: torch.Tensor, num_experts: int):
        if not self._enabled:
            return
        self._num_experts = num_experts
        layer_idx = self._layer_counter
        self._layer_counter += 1
        if layer_idx not in self._layer_data:
            self._layer_data[layer_idx] = []
        self._layer_data[layer_idx].append(top_indices.detach())

    def reset_layer_counter(self):
        """Reset layer counter at the start of each microbatch/forward pass."""
        self._layer_counter = 0

    def get_metrics_and_reset(self, step: int = 0) -> dict[str, float]:
        if not self._enabled or not self._layer_data or self._num_experts is None:
            self._layer_data.clear()
            self._layer_counter = 0
            return {}

        from slime.utils.expert_balance import compute_expert_balance_from_tensor, save_training_expert_counts

        import numpy as np

        log_dict: dict[str, float] = {}
        layer_cvs = []
        layer_lbrs = []
        layer_entropies = []

        # Prepare numpy arrays for raw count saving
        layer_numpy_data: dict[int, np.ndarray] = {}

        for layer_idx in sorted(self._layer_data.keys()):
            # Concatenate all microbatch tensors for this layer
            combined = torch.cat(self._layer_data[layer_idx], dim=0)
            metrics = compute_expert_balance_from_tensor(combined, self._num_experts, layer_idx)
            for key, val in metrics.items():
                log_dict[f"train_moe_balance/layer_{layer_idx}/{key}"] = val
            layer_cvs.append(metrics["cv"])
            layer_lbrs.append(metrics["load_balance_ratio"])
            layer_entropies.append(metrics.get("normalized_entropy", 0.0))
            layer_numpy_data[layer_idx] = combined.detach().cpu().numpy()

        if layer_cvs:
            log_dict["train_moe_balance/mean_cv"] = round(float(np.mean(layer_cvs)), 4)
            log_dict["train_moe_balance/max_cv"] = round(float(np.max(layer_cvs)), 4)
            log_dict["train_moe_balance/mean_load_balance_ratio"] = round(float(np.mean(layer_lbrs)), 4)
            log_dict["train_moe_balance/mean_normalized_entropy"] = round(float(np.mean(layer_entropies)), 4)

        # Save raw expert counts to disk for offline visualization
        save_training_expert_counts(layer_numpy_data, self._num_experts, step)

        self._layer_data.clear()
        self._layer_counter = 0
        return log_dict


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                ROUTING_REPLAY.record(top_indices)
                # Track expert balance during training
                num_experts = scores.shape[1]
                TrainingExpertBalanceTracker.get_instance().record(top_indices, num_experts)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            # Track expert balance even without routing replay
            if os.environ.get("MOE_BALANCE_TRACKING", "0") == "1":
                num_experts = scores.shape[1]
                TrainingExpertBalanceTracker.get_instance().record(top_indices, num_experts)
            return probs, top_indices

    return compute_topk


def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)
