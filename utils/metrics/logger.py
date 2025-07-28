import csv
from pathlib import Path
from typing import Dict, Union
import numpy as np

from torch.utils.tensorboard import SummaryWriter


# class MetricLogger:
#     """
#     Logs training metrics to CSV and TensorBoard inside an experiment-specific folder.
#     """
#     def __init__(self,
#                  output_dir: Union[str, Path],
#                  experiment_name: str = "experiment",
#                  use_tensorboard: bool = True):
#         self.experiment_dir = Path(output_dir) / experiment_name
#         self.experiment_dir.mkdir(parents=True, exist_ok=True)

#         self.csv_path = self.experiment_dir / f"{experiment_name}.csv"
#         self.writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard")) if use_tensorboard else None

#         self._init_csv_header = False
#         self._csv_file = open(self.csv_path, "a", newline='')
#         self._csv_writer = None

#     def log(self, metrics: Dict[str, float], step: int):
#         """
#         Logs a dict of metrics at a given step.
#         """
#         if not self._init_csv_header:
#             self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=["step"] + list(metrics.keys()))
#             if self._csv_file.tell() == 0:
#                 self._csv_writer.writeheader()
#             self._init_csv_header = True

#         row = {"step": step, **metrics}
#         self._csv_writer.writerow(row)
#         self._csv_file.flush()

#         if self.writer:
#             for k, v in metrics.items():
#                 self.writer.add_scalar(k, v, global_step=step)

#     def close(self):
#         """
#         Closes file handles and TensorBoard writer.
#         """
#         if self.writer:
#             self.writer.close()
#         self._csv_file.close()

#     def __del__(self):
#         self.close()


import csv
from pathlib import Path
from typing import Dict, Union

from torch.utils.tensorboard import SummaryWriter
import numpy as np


class MetricLogger:
    """
    Logs metrics to CSV and TensorBoard inside an experiment-specific folder.
    Supports separate CSVs for different metric groups (e.g., training, final),
    while sharing the same TensorBoard writer.
    """
    _shared_writer = None

    def __init__(self,
                 metrics_dir: Union[str, Path],
                 csv_name: str = "metrics.csv",
                 tensorboard_subdir: str = "tensorboard",
                 use_tensorboard: bool = True):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.metrics_dir / csv_name

        if use_tensorboard:
            if MetricLogger._shared_writer is None:
                tb_dir = self.metrics_dir / tensorboard_subdir
                MetricLogger._shared_writer = SummaryWriter(log_dir=str(tb_dir))
            self.writer = MetricLogger._shared_writer
        else:
            self.writer = None

        self._init_csv_header = False
        self._csv_file = open(self.csv_path, "a", newline='')
        self._csv_writer = None

    def log(self, metrics: Dict[str, float], step: int):
        """
        Logs a dict of metrics at a given step.
        Scalars go to CSV and TensorBoard; lists/arrays go only to CSV.
        """
        row = {"step": step, **metrics}

        if not self._init_csv_header:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=row.keys())
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()
            self._init_csv_header = True

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self.writer:
            for k, v in metrics.items():
                if np.isscalar(v):
                    self.writer.add_scalar(k, v, global_step=step)

    def close(self):
        if self.writer and MetricLogger._shared_writer == self.writer:
            self.writer.close()
            MetricLogger._shared_writer = None
        self._csv_file.close()

    def __del__(self):
        self.close()




def print_metrics(metrics: Dict[str, float], step: int, prefix: str = ""):
    """
    Prints metrics to stdout in a readable format.
    """
    msg = f"[Step {step}] " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(prefix + msg)
