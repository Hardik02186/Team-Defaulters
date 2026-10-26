#    pip install -U huggingface_hub
#    huggingface-cli login      # newer versions: hf auth login


from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B", local_dir="C:\\Users\\Sid\\Desktop\\Team-Defaulters\\models\\qwen3-4b")