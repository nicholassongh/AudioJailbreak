# Link: https://github.com/ictnlp/LLaMA-Omni — see that project's README.md for deployment instructions


# '_flash_supports_window_size' is not defined — this error is a _flash_attention_ version issue
# https://github.com/ictnlp/LLaMA-Omni/issues/32 has a solution

# Solution (choose the wheel file matching your torch, cuda, python, and cxx11abi versions):
# Check cxx11abi version:
# (import torch
#  print(torch._C._GLIBCXX_USE_CXX11_ABI)
# )
# ## 1. Download the wheel file
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# # 2. Install the downloaded wheel file
# pip install flash_attn-2.7.1.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Run command inside the LLaMA-Omni directory:
# bash omni_speech/infer/run.sh omni_speech/infer/examples /mnt/data/huggingface/transformers/models/Llama-3.1-8B-Omni

# Main changes are in inference/LLaMA-Omni/omni_speech/infer/examples/
# Run to_json.py to convert the JSONL file into the model's required JSON format (named question.json)
# Run the command above to perform inference — see run.sh for input/output filenames
# Run to_result.py to convert answer.json back to the final JSONL file
