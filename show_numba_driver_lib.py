# show_numba_driver_lib.py
import sys
print("Python:", sys.version.replace("\n", " "))
try:
    import numba
    from numba import cuda
    print("numba", numba.__version__)
    print("cuda.is_available()", cuda.is_available())
    # print underlying driver lib object
    import inspect
    try:
        lib = cuda.cudadrv.driver.lib
        print("numba driver lib object:", lib)
        try:
            name = lib._name
        except Exception:
            # fallback
            name = getattr(lib, '__file__', repr(lib))
        print("driver lib._name (or fallback):", name)
    except Exception as e:
        print("Could not access cuda.cudadrv.driver.lib:", e)
except Exception as e:
    print("Import error:", e)
    raise
