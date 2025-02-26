import sys


def trace_module(frame, event, arg):
    code = frame.f_code
    module_name = code.co_filename
    if (
        event == "call"
        and ".pyenv" not in module_name
        and ".cache" not in module_name
        and "<" not in module_name
    ):
        print(f"Module loaded: {module_name}")
    return trace_module


sys.settrace(trace_module)
