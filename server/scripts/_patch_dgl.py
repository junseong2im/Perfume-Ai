"""Patch DGL graphbolt to skip if DLL not found"""
import os, sys

gb_init = os.path.join(sys.prefix, 'Lib', 'site-packages', 'dgl', 'graphbolt', '__init__.py')
print(f"Patching: {gb_init}")

with open(gb_init, 'r') as f:
    content = f.read()

if 'Patched' in content:
    print("Already patched")
else:
    lines = content.split('\n')
    indented = '\n'.join('    ' + line for line in lines)
    new_content = (
        '# Patched to gracefully skip if DLL not found\n'
        'try:\n'
        + indented + '\n'
        'except (FileNotFoundError, ImportError, OSError) as e:\n'
        '    import warnings\n'
        '    warnings.warn(f"DGL graphbolt not available: {e}")\n'
    )
    with open(gb_init, 'w') as f:
        f.write(new_content)
    print("Patched OK")
