Place complete local Hugging Face snapshots here or use absolute paths in the GUI.

- MOSS-TTS-v1.5: https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5/tree/main
- MOSS-Audio-Tokenizer: https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer/tree/main

Download all safetensors shards, their index, JSON files, Python files and relative
imports, tokenizer and chat template. Preserve subfolders. Do not substitute a
LoRA/GGUF/Realtime model or the v2 codec. The 8B v1.5 model uses the original 24kHz codec.
Local Python is executed with trust_remote_code=True; use trusted snapshots.
