import torch
import comfy.model_management
from ..core import logger
import os
import platform

class CGPUInfo:
    def __init__(self):
        self.pynvmlLoaded = False
        self.amdgpuLoaded = False
        self.anygpuLoaded = False
        self.cudaDevicesFound = 0
        self.gpus = []

        # Tentar NVIDIA
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.pynvmlLoaded = True
        except:
            pass

        # Tentar AMD se não for NVIDIA
        if not self.pynvmlLoaded:
            try:
                import pyamdgpuinfo
                if pyamdgpuinfo.detect_gpus() > 0:
                    self.pyamdgpu = pyamdgpuinfo
                    self.amdgpuLoaded = True
            except:
                pass

        self.anygpuLoaded = self.pynvmlLoaded or self.amdgpuLoaded
        
        if self.anygpuLoaded:
            count = self.deviceGetCount()
            self.cudaDevicesFound = count
            for i in range(count):
                handle = self.deviceGetHandleByIndex(i)
                self.gpus.append({'index': i, 'name': self.deviceGetName(handle, i)})

    def deviceGetCount(self):
        if self.pynvmlLoaded: return self.pynvml.nvmlDeviceGetCount()
        if self.amdgpuLoaded: return self.pyamdgpu.detect_gpus()
        return 0

    def deviceGetHandleByIndex(self, index):
        if self.pynvmlLoaded: return self.pynvml.nvmlDeviceGetHandleByIndex(index)
        if self.amdgpuLoaded: return self.pyamdgpu.get_gpu(index)
        return 0

    def deviceGetName(self, handle, index):
        if self.pynvmlLoaded:
            n = self.pynvml.nvmlDeviceGetName(handle)
            return n.decode('utf-8') if isinstance(n, bytes) else n
        return "AMD GPU"

    def deviceGetMemoryInfo(self, handle):
        # Prioridade para Torch em AMD (evita travar em 100%)
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(handle if isinstance(handle, int) else 0)
                return {'total': total, 'used': total - free}
            except: pass
        
        if self.pynvmlLoaded:
            m = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {'total': m.total, 'used': m.used}
        elif self.amdgpuLoaded:
            return {'total': handle.vram_size, 'used': handle.query_vram_usage()}
        return {'total': 1, 'used': 0}

    def getStatus(self):
        gpus_data = []
        if not self.anygpuLoaded:
            return {'device_type': 'cpu', 'gpus': []}

        for i in range(self.cudaDevicesFound):
            h = self.deviceGetHandleByIndex(i)
            mem = self.deviceGetMemoryInfo(h)
            util = 0
            temp = 0
            try:
                if self.pynvmlLoaded: util = self.pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                elif self.amdgpuLoaded: util = int(h.query_load() * 100)
                if self.pynvmlLoaded: temp = self.pynvml.nvmlDeviceGetTemperature(h, 0)
                elif self.amdgpuLoaded: temp = h.query_temperature()
            except: pass

            gpus_data.append({
                'gpu_utilization': util,
                'gpu_temperature': temp,
                'vram_total': mem['total'],
                'vram_used': mem['used'],
                'vram_used_percent': (mem['used'] / mem['total'] * 100) if mem['total'] > 0 else 0,
            })
        return {'device_type': 'cuda', 'gpus': gpus_data}

    def getInfo(self): return self.gpus
