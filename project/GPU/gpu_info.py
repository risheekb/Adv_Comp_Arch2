
import pynvml

# Initialize NVIDIA Management Library
pynvml.nvmlInit()

# Get GPU count
device_count = pynvml.nvmlDeviceGetCount()

# CUDA cores per SM for various architectures
CUDA_CORES_PER_SM = {
    "Kepler": 192,
    "Maxwell": 128,
    "Pascal": 64,
    "Volta": 64,
    "Turing": 64,
    "Ampere": 128,
    "Hopper": 128
}

def get_cuda_cores(handle):
    """Estimate CUDA cores based on the GPU architecture."""
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        gpu_name = pynvml.nvmlDeviceGetName(handle)  # No .decode() needed!

        # Get the estimated CUDA cores per SM based on the architecture
        for arch, cores_per_sm in CUDA_CORES_PER_SM.items():
            if arch in gpu_name:
                return major * minor * cores_per_sm  # Estimate based on Compute Capability

        return major * minor * 64  # Default assumption (Pascal-based)

    except pynvml.NVMLError as e:
        print(f"Error fetching CUDA cores: {e}")
        return "Unknown"

# Iterate through each GPU and print CUDA core details
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    gpu_name = pynvml.nvmlDeviceGetName(handle)  # No .decode() needed!
    total_cuda_cores = get_cuda_cores(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # GPU usage in %

    print(f"GPU {i}: {gpu_name}")
    print(f"Total CUDA Cores (Estimated): {total_cuda_cores}")
    print(f"CUDA Cores Utilized: ~{int(total_cuda_cores * (utilization / 100))} cores")
    print("=" * 40)

# Shutdown NVML
pynvml.nvmlShutdown()

