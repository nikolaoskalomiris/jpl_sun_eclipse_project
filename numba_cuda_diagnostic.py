# numba_cuda_diagnostic.py
import sys, traceback
print("Python:", sys.version.replace("\n", " "))
try:
    import numba
    from numba import cuda
    print("numba:", numba.__version__)
except Exception as e:
    print("ERROR importing numba/cuda:", e)
    traceback.print_exc()
    sys.exit(1)

print("cuda.is_available():", cuda.is_available())

# list devices (safe)
try:
    devs = list(cuda.gpus)
    print("cuda.gpus list length:", len(devs))
    for i, d in enumerate(devs):
        try:
            print("device", i, "repr:", repr(d))
            # device attributes (some work even if context not created)
            try:
                attrs = d.get_attributes()
                print("  attrs keys:", list(attrs.keys()))
            except Exception as e:
                print("  get_attributes() error:", e)
        except Exception as e:
            print("  device listing error:", e)
except Exception as e:
    print("Error enumerating cuda.gpus:", e)

# Try to detect (this prints info and may raise)
try:
    print("Calling cuda.detect() ...")
    cuda.detect()   # prints device detect; may raise on failure
except Exception as e:
    print("cuda.detect() raised:", repr(e))

# Attempt to initialize driver and create a context (catch exceptions)
try:
    print("Attempting to initialize driver and create a context...")
    try:
        # explicit init (low-level)
        from numba.cuda.cudadrv import driver as cudadrv
        cudadrv.init()
        ver = cudadrv.get_version()
        print("Driver version (numba):", ver)
    except Exception as e:
        print("cudadrv.init/get_version error:", repr(e))
    # attempt current_context() - this creates/gets context
    try:
        ctx = cuda.current_context()
        print("current_context() OK:", ctx)
    except Exception as e:
        print("current_context() error:", repr(e))
    # attempt a tiny allocation
    try:
        d = cuda.device_array(4)
        print("device_array allocated OK:", type(d), "size:", d.size)
        d.copy_to_host()
        print("copy_to_host OK")
    except Exception as e:
        print("tiny device allocation error:", repr(e))
except Exception as e:
    print("Driver/context level failure:", repr(e))
    traceback.print_exc()

# simple test invocation of driver API if possible
try:
    from numba.cuda.cudadrv import drvapi
    print("drvapi module imported")
except Exception as e:
    print("drvapi import failed:", e)

print("DONE diagnostic.")
