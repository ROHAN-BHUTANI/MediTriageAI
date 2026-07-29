"""Hooks for integrating E-PATH-CO-REASON with training and evaluation scripts."""

from __future__ import annotations

import time
import json
from pathlib import Path
import torch
import torch.nn as nn


def apply_loss_hook(
    model: nn.Module,
    specialist_logits: torch.Tensor,
    severity_logits: torch.Tensor,
    labels_specialist: torch.Tensor,
    labels_severity: torch.Tensor,
    loss_fn: nn.Module,
) -> dict[str, torch.Tensor]:
    """Intercept loss computation and delegate to E-PATH-CO-REASON custom loss solver.
    
    If the model does not support a custom loss, defaults to standard joint loss evaluation.
    """
    if hasattr(model, "compute_loss"):
        return model.compute_loss(
            specialist_logits,
            severity_logits,
            labels_specialist,
            labels_severity,
            loss_fn,
        )
    return loss_fn(specialist_logits, severity_logits, labels_specialist, labels_severity)


class ExecutionEngineAuditor:
    """Observability Auditor for ReasoningPathExecutionEngine.
    
    Records shapes, data types, device information, statistical properties,
    memory metrics, and execution times using PyTorch hooks.
    """

    def __init__(self, engine: nn.Module) -> None:
        self.engine = engine
        self.reset()
        
        # Register module-level hooks
        self.forward_pre_hook_handle = engine.register_forward_pre_hook(self.forward_pre_hook)
        self.forward_hook_handle = engine.register_forward_hook(self.forward_hook)

        # Handles to cleanup block hooks dynamically
        self.block_hook_handles = []

    def reset(self) -> None:
        """Reset all metric accumulators."""
        self.inputs_audit: list[dict] = []
        self.outputs_audit: list[dict] = []
        self.activations_audit: list[dict] = []
        self.memory_audit: list[dict] = []
        self.timings_audit: list[dict] = []
        
        # Temporary tracking states
        self.current_step = 0
        self.current_forward_start_time = 0.0
        self.current_memory_before: dict = {}
        self.current_memory_after: dict = {}
        self.block_activations: dict = {}

    def set_current_step(self, step: int) -> None:
        """Update current step index of the execution path."""
        self.current_step = step

    def _get_tensor_stats(self, tensor: torch.Tensor) -> dict:
        """Calculate shape, dtype, device, mean, std, min, max of a tensor."""
        t_detached = tensor.detach()
        return {
            "shape": list(t_detached.shape),
            "dtype": str(t_detached.dtype),
            "device": str(t_detached.device),
            "mean": float(t_detached.mean().item()),
            "std": float(t_detached.std().item()) if t_detached.numel() > 1 else 0.0,
            "min": float(t_detached.min().item()),
            "max": float(t_detached.max().item())
        }

    def _get_memory_usage(self, device: torch.device) -> dict:
        """Retrieve memory stats for GPU/CPU."""
        if device.type == "cuda":
            return {
                "memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
                "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            }
        else:
            try:
                import psutil
                process = psutil.Process()
                return {
                    "memory_allocated_mb": process.memory_info().rss / (1024 ** 2),
                    "max_memory_allocated_mb": process.memory_info().peak_wset / (1024 ** 2) if hasattr(process.memory_info(), "peak_wset") else process.memory_info().rss / (1024 ** 2)
                }
            except Exception:
                return {
                    "memory_allocated_mb": 0.0,
                    "max_memory_allocated_mb": 0.0
                }

    def forward_pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Executes before ReasoningPathExecutionEngine forward pass."""
        # args[0]: evidence_list, args[1]: routing_decision, args[2]: blocks
        evidence_list = args[0]
        routing_decision = args[1]
        blocks = args[2]

        device = evidence_list[0].device
        self.current_step = 0
        self.block_activations = {}
        
        # 1. Capture inputs (Item 1)
        batch_idx = len(self.inputs_audit)
        batch_input_stats = {
            "batch_index": batch_idx,
            "timestamp": time.time(),
        }
        aspect_names = ["symptom", "anatomical", "temporal", "systemic"]
        for idx, tensor in enumerate(evidence_list):
            name = aspect_names[idx]
            batch_input_stats[name] = self._get_tensor_stats(tensor)
        self.inputs_audit.append(batch_input_stats)

        # 2. Capture memory before (Item 6)
        self.current_memory_before = self._get_memory_usage(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # 3. Capture forward start time (Item 7)
        if device.type == "cuda":
            torch.cuda.synchronize()
        self.current_forward_start_time = time.perf_counter()

        # 4. Attach block-level activation hooks (Item 5)
        self._clear_block_hooks()
        for b_idx, block in enumerate(blocks):
            handle_pre = block.register_forward_pre_hook(self._make_block_pre_hook(b_idx))
            handle_post = block.register_forward_hook(self._make_block_post_hook(b_idx))
            self.block_hook_handles.extend([handle_pre, handle_post])

    def _make_block_pre_hook(self, block_idx: int):
        def hook(module: nn.Module, args: tuple) -> None:
            input_tensor = args[0]
            key = f"step_{self.current_step}_block_{block_idx}"
            if key not in self.block_activations:
                self.block_activations[key] = {}
            self.block_activations[key]["before"] = self._get_tensor_stats(input_tensor)
        return hook

    def _make_block_post_hook(self, block_idx: int):
        def hook(module: nn.Module, args: tuple, output_tensor: torch.Tensor) -> None:
            key = f"step_{self.current_step}_block_{block_idx}"
            if key not in self.block_activations:
                self.block_activations[key] = {}
            self.block_activations[key]["after"] = self._get_tensor_stats(output_tensor)
        return hook

    def _clear_block_hooks(self) -> None:
        for handle in self.block_hook_handles:
            handle.remove()
        self.block_hook_handles = []

    def forward_hook(self, module: nn.Module, args: tuple, output: tuple) -> None:
        """Executes after ReasoningPathExecutionEngine forward pass."""
        final_state, thought_path = output
        device = final_state.device

        # 1. Capture forward execution time (Item 7)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - self.current_forward_start_time

        # 2. Capture memory after and peak GPU memory (Item 6)
        self.current_memory_after = self._get_memory_usage(device)
        peak_gpu = 0.0
        if device.type == "cuda":
            peak_gpu = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        batch_idx = len(self.memory_audit)
        self.memory_audit.append({
            "batch_index": batch_idx,
            "before_mb": self.current_memory_before["memory_allocated_mb"],
            "after_mb": self.current_memory_after["memory_allocated_mb"],
            "peak_gpu_mb": peak_gpu,
            "peak_system_mb": self.current_memory_before["max_memory_allocated_mb"],
        })

        # 3. Capture output (Item 2)
        self.outputs_audit.append({
            "batch_index": batch_idx,
            "final_state": self._get_tensor_stats(final_state)
        })

        # 4. Save block activation stats (Item 5)
        self.activations_audit.append({
            "batch_index": batch_idx,
            "blocks": self.block_activations.copy()
        })

        # 5. Save timing stats (Item 7)
        self.timings_audit.append({
            "batch_index": batch_idx,
            "forward_execution_time_seconds": forward_time,
            "backward_execution_time_seconds": 0.0
        })

        # Clean block hooks
        self._clear_block_hooks()

    def finalize_and_export_audit(
        self,
        model: nn.Module,
        last_batch: dict | None,
        device: torch.device,
        use_amp: bool,
        checkpoint_dir: str | Path | None
    ) -> None:
        """Executes standalone backward pass instrumentation and exports JSON logs and MD summary."""
        backward_time = 0.0
        grad_norms = {}
        layer_grad_norms = {}
        total_grad_norm = 0.0
        received_gradients = {}
        
        # 1. Standing backward pass audit (Items 3, 7, 9)
        if last_batch is not None:
            model.eval()
            model.zero_grad()
            
            # Prepare inputs
            input_ids = last_batch["input_ids"].to(device)
            attention_mask = last_batch["attention_mask"].to(device)
            labels_spec = last_batch["labels_specialist"].to(device)
            labels_sev = last_batch["labels_severity"].to(device)
            
            with torch.enable_grad():
                device_type = "cuda" if device.type == "cuda" else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    outputs = model(input_ids, attention_mask)
                    
                    from src.model import JointLoss
                    loss_fn = JointLoss()
                    loss_dict = apply_loss_hook(
                        model,
                        outputs.specialist_logits,
                        outputs.severity_logits,
                        labels_spec,
                        labels_sev,
                        loss_fn
                    )
                    loss = loss_dict["joint_loss"]

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                
                loss.backward()
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                backward_time = time.perf_counter() - t_start

            # Capture gradients (Item 3, 9)
            total_grad_sum_sq = 0.0
            for b_idx, block in enumerate(model.blocks):
                block_grad_sum_sq = 0.0
                for p_name, p in block.named_parameters():
                    full_name = f"blocks.{b_idx}.{p_name}"
                    if p.requires_grad:
                        if p.grad is not None:
                            p_grad_norm = float(p.grad.norm().item())
                            grad_norms[full_name] = p_grad_norm
                            block_grad_sum_sq += p_grad_norm ** 2
                            total_grad_sum_sq += p_grad_norm ** 2
                            received_gradients[full_name] = bool(p_grad_norm > 1e-9)
                        else:
                            grad_norms[full_name] = 0.0
                            received_gradients[full_name] = False
                    else:
                        grad_norms[full_name] = 0.0
                        received_gradients[full_name] = False
                
                layer_grad_norms[f"blocks.{b_idx}"] = float(block_grad_sum_sq ** 0.5)
                
            total_grad_norm = float(total_grad_sum_sq ** 0.5)
            model.zero_grad()

        if self.timings_audit:
            self.timings_audit[-1]["backward_execution_time_seconds"] = backward_time

        # 2. Capture parameter statistics (Item 4, 8)
        parameter_statistics = {}
        for b_idx, block in enumerate(model.blocks):
            for p_name, p in block.named_parameters():
                full_name = f"blocks.{b_idx}.{p_name}"
                p_data = p.detach()
                nan_check = bool(torch.isnan(p_data).any())
                inf_check = bool(torch.isinf(p_data).any())
                
                parameter_statistics[full_name] = {
                    "mean": float(p_data.mean().item()),
                    "std": float(p_data.std().item()),
                    "min": float(p_data.min().item()),
                    "max": float(p_data.max().item()),
                    "NaNs": nan_check,
                    "Inf": inf_check,
                    "require_grad": bool(p.requires_grad)
                }

        # 3. Format and save everything
        audit_data = {
            "inputs": self.inputs_audit,
            "outputs": self.outputs_audit
        }
        gradients_data = {
            "gradient_norms_per_parameter": grad_norms,
            "gradient_norms_per_layer": layer_grad_norms,
            "total_gradient_norm": total_grad_norm,
            "received_gradients": received_gradients
        }
        statistics_data = {
            "parameter_statistics": parameter_statistics
        }
        activations_data = {
            "activations": self.activations_audit
        }
        memory_data = {
            "memory_usage": self.memory_audit
        }
        timing_data = {
            "timings": self.timings_audit
        }

        # Resolve output directories
        export_dirs = [Path(".")]
        if checkpoint_dir is not None:
            export_dirs.append(Path(checkpoint_dir))

        for export_dir in export_dirs:
            export_dir.mkdir(parents=True, exist_ok=True)
            
            with open(export_dir / "execution_engine_audit.json", "w", encoding="utf-8") as f:
                json.dump(audit_data, f, indent=4)
            with open(export_dir / "execution_engine_gradients.json", "w", encoding="utf-8") as f:
                json.dump(gradients_data, f, indent=4)
            with open(export_dir / "execution_engine_statistics.json", "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=4)
            with open(export_dir / "execution_engine_activations.json", "w", encoding="utf-8") as f:
                json.dump(activations_data, f, indent=4)
            with open(export_dir / "execution_engine_memory.json", "w", encoding="utf-8") as f:
                json.dump(memory_data, f, indent=4)
            with open(export_dir / "execution_engine_timing.json", "w", encoding="utf-8") as f:
                json.dump(timing_data, f, indent=4)

            self.write_markdown_summary(export_dir, audit_data, gradients_data, statistics_data, activations_data, memory_data, timing_data)

    def write_markdown_summary(
        self,
        export_dir: Path,
        audit_data: dict,
        gradients_data: dict,
        statistics_data: dict,
        activations_data: dict,
        memory_data: dict,
        timing_data: dict
    ) -> None:
        """Generates a human-readable markdown report of the execution engine audit."""
        # Calculate summary statistics
        num_batches = len(audit_data["inputs"])
        total_p_grad_norm = gradients_data["total_gradient_norm"]
        
        # Calculate avg forward time
        avg_fwd = sum(t["forward_execution_time_seconds"] for t in timing_data["timings"]) / num_batches if num_batches > 0 else 0.0
        bwd_time = timing_data["timings"][-1]["backward_execution_time_seconds"] if timing_data["timings"] else 0.0
        
        # Peak GPU memory
        peak_gpu_mem = max((m["peak_gpu_mb"] for m in memory_data["memory_usage"]), default=0.0)
        
        # Parameter counts
        total_params = len(statistics_data["parameter_statistics"])
        nan_params = sum(1 for p in statistics_data["parameter_statistics"].values() if p["NaNs"])
        inf_params = sum(1 for p in statistics_data["parameter_statistics"].values() if p["Inf"])
        req_grad_params = sum(1 for p in statistics_data["parameter_statistics"].values() if p["require_grad"])
        received_grad_params = sum(1 for v in gradients_data["received_gradients"].values() if v)

        # Build markdown content
        lines = [
            "# E-PATH-CO-REASON Execution Engine Observability Report",
            "",
            "## Summary Metrics",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Validation Batches Audited** | {num_batches} |",
            f"| **Average Forward Execution Time** | {avg_fwd * 1000:.3f} ms |",
            f"| **Backward Execution Time (Instrumented Batch)** | {bwd_time * 1000:.3f} ms |",
            f"| **Peak GPU Memory Allocation** | {peak_gpu_mem:.2f} MB |",
            f"| **Total Tracked Engine Parameters** | {total_params} |",
            f"| **Parameters Requiring Gradient** | {req_grad_params} |",
            f"| **Parameters Actually Receiving Gradients** | {received_grad_params} |",
            f"| **Total Gradient Norm** | {total_p_grad_norm:.6f} |",
            f"| **Parameter NaNs / Infs Detected** | {nan_params} / {inf_params} |",
            "",
            "## Input / Output Tensors Statistics",
            "### Inputs (evidence_list aspect embeddings)",
        ]

        if num_batches > 0:
            last_in = audit_data["inputs"][-1]
            lines.extend([
                "| Input Tensor | Shape | Dtype | Device | Mean | Std | Min | Max |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
            ])
            for name in ["symptom", "anatomical", "temporal", "systemic"]:
                if name in last_in:
                    t = last_in[name]
                    lines.append(f"| `{name}` | {t['shape']} | `{t['dtype']}` | `{t['device']}` | {t['mean']:.6f} | {t['std']:.6f} | {t['min']:.6f} | {t['max']:.6f} |")
        else:
            lines.append("No batches audited.")

        lines.extend([
            "",
            "### Outputs (final_state latent representation)",
        ])
        
        if num_batches > 0:
            last_out = audit_data["outputs"][-1]
            lines.extend([
                "| Tensor | Shape | Dtype | Device | Mean | Std | Min | Max |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
            ])
            t = last_out["final_state"]
            lines.append(f"| `final_state` | {t['shape']} | `{t['dtype']}` | `{t['device']}` | {t['mean']:.6f} | {t['std']:.6f} | {t['min']:.6f} | {t['max']:.6f} |")
        else:
            lines.append("No batches audited.")

        lines.extend([
            "",
            "## Gradient Norms per Layer",
            "| Layer | Gradient Norm |",
            "| :--- | :--- |"
        ])
        for layer, val in gradients_data["gradient_norms_per_layer"].items():
            lines.append(f"| `{layer}` | {val:.6f} |")

        lines.extend([
            "",
            "## Step-wise Activations (Pre-Forward vs Post-Forward)",
            "| Step / Executed Block | Activation Phase | Shape | Mean | Std | Min | Max |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        ])

        if activations_data["activations"]:
            last_acts = activations_data["activations"][-1]["blocks"]
            for key in sorted(last_acts.keys()):
                act = last_acts[key]
                if "before" in act:
                    b = act["before"]
                    lines.append(f"| `{key}` | Before Forward | {b['shape']} | {b['mean']:.6f} | {b['std']:.6f} | {b['min']:.6f} | {b['max']:.6f} |")
                if "after" in act:
                    a = act["after"]
                    lines.append(f"| `{key}` | After Forward | {a['shape']} | {a['mean']:.6f} | {a['std']:.6f} | {a['min']:.6f} | {a['max']:.6f} |")
        else:
            lines.append("No activations tracked.")

        summary_file = export_dir / "execution_engine_summary.md"
        summary_file.write_text("\n".join(lines), encoding="utf-8")
