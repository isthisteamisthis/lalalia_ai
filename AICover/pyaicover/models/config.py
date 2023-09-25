import sys
import torch
from multiprocessing import cpu_count

def use_fp32_config():
    for config_file in ["32k.json", "40k.json", "48k.json"]:
        with open(f"configs/{config_file}", "r") as f:
            strr = f.read().replace("true", "false")
        with open(f"configs/{config_file}", "w") as f:
            f.write(strr)
    with open("trainset_preprocess_pipeline_print.py", "r") as f:
        strr = f.read().replace("3.7", "3.0")
    with open("trainset_preprocess_pipeline_print.py", "w") as f:
        f.write(strr)


class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.python_cmd = sys.executable or "python"
        self.listen_port = 8888
        self.iscolab = False
        self.noparallel = False
        self.noautoopen = False
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                self.is_half = False
                use_fp32_config()
            else:
                pass
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif self.has_mps():
            print("No supported Nvidia GPU found, use MPS instead")
            self.device = "mps"
            self.is_half = False
            use_fp32_config()
        else:
            print("No supported Nvidia GPU found, use CPU instead")
            self.device = "cpu"
            self.is_half = False
            use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max
