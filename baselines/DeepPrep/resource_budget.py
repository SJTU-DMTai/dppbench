"""Multi-GPU accounting selected only by LLM experiments."""
from baselines.common.budget import Budget


class LLMResourceBudget(Budget):
    def start(self):
        super().start()
        import torch
        if torch.cuda.is_available():
            for index in range(1, torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)

    def violations(self):
        errors = super().violations()
        import torch
        if torch.cuda.is_available():
            peak = sum(max(torch.cuda.max_memory_allocated(i),
                           torch.cuda.max_memory_reserved(i))
                       for i in range(torch.cuda.device_count())) / 1024**2
            self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, peak)
            if self.max_gpu_memory_mb is not None and self.peak_gpu_memory_mb > self.max_gpu_memory_mb:
                if not any(e.startswith('GPU memory budget exceeded:') for e in errors):
                    errors.append(f'GPU memory budget exceeded: {self.peak_gpu_memory_mb:.1f}MB/{self.max_gpu_memory_mb:.1f}MB')
        return errors
