import torch
import comfy.model_management
from ..core import logger
import os
import platform

def is_jetson() -> bool:
    """
    Determines if the Python environment is running on a Jetson device.
    """
    PROC_DEVICE_MODEL = ''
    try:
        with open('/proc/device-tree/model', 'r') as f:
            PROC_DEVICE_MODEL = f.read().strip()
            logger.info(f"Device model: {PROC_DEVICE_MODEL}")
            return "NVIDIA" in PROC_DEVICE_MODEL
    except Exception:
        platform_release = platform.release()
        if 'tegra' in platform_release.lower():
            return True
        else:
            return False

IS_JETSON = is_jetson()

class CGPUInfo:
    """
    This class is responsible for getting information from GPU (NVIDIA, AMD, or Jetson).
    """
    cuda = False
    pynvmlLoaded = False
    jtopLoaded = False
    amdgpuLoaded = False
    cudaAvailable = False
    torchDevice = 'cpu'
    cudaDevice = 'cpu'
    cudaDevicesFound = 0
    switchGPU = True
    switchVRAM = True
    switchTemperature = True
    gpus = []
    gpusUtilization = []
    gpusVRAM = []
    gpusTemperature = []

    def __init__(self):
        if IS_JETSON:
            try:
                from jtop import jtop
                self.jtopInstance = jtop()
                self.jtopInstance.start()
                self.jtopLoaded = True
                logger.info('Crystools: jtop initialized (Jetson).')
            except Exception as e:
                logger.error('Crystools: Could not init jtop. ' + str(e))
        else:
            # 1. Tentar NVIDIA
            try:
                import pynvml
                self.pynvml = pynvml
                self.pynvml.nvmlInit()
                self.pynvmlLoaded = True
                logger.info('Crystools: pynvml (NVIDIA) initialized.')
            except Exception:
                # 2. Se falhar NVIDIA, tentar AMD
                try:
                    import pyamdgpuinfo
                    if pyamdgpuinfo.detect_gpus() > 0:
                        self.pyamdgpu = pyamdgpuinfo
                        self.amdgpuLoaded = True
                        logger.info('Crystools: pyamdgpuinfo (AMD) initialized.')
                except Exception as e:
                    logger.info('Crystools: Neither NVIDIA nor AMD monitoring tools found.')

        self.anygpuLoaded = self.pynvmlLoaded or self.jtopLoaded or self.amdgpuLoaded

        try:
            self.torchDevice = comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device())
        except Exception as e:
            logger.error('Crystools: Could not pick default device. ' + str(e))

        if self.anygpuLoaded:
            count = self.deviceGetCount()
            if count > 0:
                self.cudaDevicesFound = count
                logger.info(f"Crystools: Detected {count} GPU(s).")

                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)
                    gpuName = self.deviceGetName(deviceHandle, deviceIndex)

                    logger.info(f"GPU {deviceIndex}: {gpuName}")

                    self.gpus.append({
                        'index': deviceIndex,
                        'name': gpuName,
                    })

                    self.gpusUtilization.append(True)
                    self.gpusVRAM.append(True)
                    self.gpusTemperature.append(True)

                self.cuda = True
                logger.info(self.systemGetDriverVersion())
            else:
                logger.warning('Crystools: No GPU detected by monitoring libraries.')
        else:
            logger.warning('Crystools: No GPU monitoring libraries available (pynvml or pyamdgpuinfo).')

        self.cudaDevice = 'cpu' if self.torchDevice == 'cpu' else 'cuda'
        self.cudaAvailable = torch.cuda.is_available()

    def deviceGetCount(self):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetCount()
        elif self.jtopLoaded:
            return 1
        elif self.amdgpuLoaded:
            return self.pyamdgpu.detect_gpus()
        return 0

    def deviceGetHandleByIndex(self, index):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetHandleByIndex(index)
        elif self.jtopLoaded:
            return index
        elif self.amdgpuLoaded:
            return self.pyamdgpu.get_gpu(index)
        return 0

    def deviceGetName(self, deviceHandle, deviceIndex):
        if self.pynvmlLoaded:
            try:
                name = self.pynvml.nvmlDeviceGetName(deviceHandle)
                return name.decode('utf-8') if isinstance(name, bytes) else name
            except:
                return 'NVIDIA GPU'
        elif self.jtopLoaded:
            try:
                return next(iter(self.jtopInstance.gpu.keys()))
            except:
                return 'Jetson GPU'
        elif self.amdgpuLoaded:
            try:
                return deviceHandle.name
            except:
                return 'AMD GPU'
        return 'Unknown GPU'

    def systemGetDriverVersion(self):
        if self.pynvmlLoaded:
            return f'Driver: NVIDIA {self.pynvml.nvmlSystemGetDriverVersion()}'
        elif self.amdgpuLoaded:
            return 'Driver: AMD (via pyamdgpuinfo)'
        return 'Driver: Unknown'

    def deviceGetUtilizationRates(self, deviceHandle):
        try:
            if self.pynvmlLoaded:
                return self.pynvml.nvmlDeviceGetUtilizationRates(deviceHandle).gpu
            elif self.jtopLoaded:
                return self.jtopInstance.stats.get('GPU', -1)
            elif self.amdgpuLoaded:
                # Retorna float 0.0-1.0, convertemos para 0-100
                return int(deviceHandle.query_load() * 100)
        except:
            return -1
        return 0

    def deviceGetMemoryInfo(self, deviceHandle):
        try:
            if self.pynvmlLoaded:
                mem = self.pynvml.nvmlDeviceGetMemoryInfo(deviceHandle)
                return {'total': mem.total, 'used': mem.used}
            elif self.jtopLoaded:
                mem_data = self.jtopInstance.memory['RAM']
                return {'total': mem_data['tot'], 'used': mem_data['used']}
            elif self.amdgpuLoaded:
                return {
                    'total': deviceHandle.vram_size,
                    'used': deviceHandle.query_vram_usage()
                }
        except:
            pass
        return {'total': 1, 'used': 1}

    def deviceGetTemperature(self, deviceHandle):
        try:
            if self.pynvmlLoaded:
                return self.pynvml.nvmlDeviceGetTemperature(deviceHandle, self.pynvml.NVML_TEMPERATURE_GPU)
            elif self.jtopLoaded:
                return self.jtopInstance.stats.get('Temp gpu', -1)
            elif self.amdgpuLoaded:
                return deviceHandle.query_temperature()
        except:
            return -1
        return 0

    def getInfo(self):
        return self.gpus

    def getStatus(self):
        gpus = []
        if self.cudaDevice == 'cpu':
            gpuType = 'cpu'
            gpus.append({
                'gpu_utilization': -1,
                'gpu_temperature': -1,
                'vram_total': -1,
                'vram_used': -1,
                'vram_used_percent': -1,
            })
        else:
            gpuType = self.cudaDevice
            for deviceIndex in range(self.cudaDevicesFound):
                deviceHandle = self.deviceGetHandleByIndex(deviceIndex)
                
                util = self.deviceGetUtilizationRates(deviceHandle)
                temp = self.deviceGetTemperature(deviceHandle)
                mem = self.deviceGetMemoryInfo(deviceHandle)
                
                vramPercent = (mem['used'] / mem['total'] * 100) if mem['total'] > 0 else 0
                
                gpus.append({
                    'gpu_utilization': util,
                    'gpu_temperature': temp,
                    'vram_total': mem['total'],
                    'vram_used': mem['used'],
                    'vram_used_percent': vramPercent,
                })

        return {
            'device_type': gpuType,
            'gpus': gpus,
        }

    def close(self):
        if self.jtopLoaded and self.jtopInstance is not None:
            self.jtopInstance.close()
