
#!/usr/bin/env bash

set -e -x
torchrun --nproc-per-node=4 --standalone tests/special_distributed/test_tensor_dict.py
torchrun --nproc-per-node=4 --standalone tests/special_distributed/test_torch_functional.py
