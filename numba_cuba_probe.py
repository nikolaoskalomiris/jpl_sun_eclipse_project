from numba import cuda
import numba
print("numba:", numba.__version__)
print("cuda.is_available()", cuda.is_available())
try:
    print("cuda.gpus:", list(cuda.gpus))
except Exception as e:
    print("cuda.gpus() error:", e)
try:
    cuda.detect()
except Exception as e:
    print("cuda.detect() exception:", e)
# try a tiny allocation:
try:
    a = cuda.device_array(8)
    print("device_array alloc OK")
except Exception as e:
    print("device_array allocation error:", e)
