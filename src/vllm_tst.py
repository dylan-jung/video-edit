import os
import sys

# Get directory path where vllm is installed
# You can find this with `which vllm` in terminal
vllm_path = "/Users/minhyeok/Documents/d[Develop] Dev/attenz-ai/attenz-ai/vllm_env/bin"  # Replace with actual path from `which vllm`
vllm_dir = "/Users/minhyeok/Documents/d[Develop] Dev/attenz-ai/attenz-ai/vllm"
sys.path.append(os.path.dirname(vllm_path))
sys.path.append(os.path.dirname(vllm_dir))
# Now imports should work
import vllm

print(vllm.__file__)
