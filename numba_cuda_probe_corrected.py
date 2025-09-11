# numba_cuda_probe_corrected.py
import sys, traceback
print("Python:", sys.version.replace("\n"," "))
try:
    import numba
    from numba import cuda
    print("numba", numba.__version__)
except Exception as e:
    print("Failed to import numba/cuda:", e)
    traceback.print_exc()
    sys.exit(1)

print("cuda.is_available():", cuda.is_available())
print("cuda.gpus:", list(cuda.gpus))

# run cuda.detect() (this prints device info)
try:
    print("\nRunning cuda.detect() ...")
    cuda.detect()
except Exception as e:
    print("cuda.detect() raised:")
    traceback.print_exc()

# Attempt to get current_context() and allocate a tiny buffer
try:
    print("\nAttempting to create/get current_context() ...")
    try:
        ctx = cuda.current_context()
        print("current_context() succeeded:", ctx)
    except Exception as e:
        print("current_context() raised:")
        traceback.print_exc()

    print("\nAttempting trivial device allocation / copy ...")
    try:
        d = cuda.device_array(8)
        print("device_array allocated:", d)
        h = d.copy_to_host()
        print("copy_to_host succeeded, len:", len(h))
    except Exception as e:
        print("device allocation/copy raised:")
        traceback.print_exc()
except Exception as e:
    print("Unexpected error:")
    traceback.print_exc()

print("\nProbe done.")
