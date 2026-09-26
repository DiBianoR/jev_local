# start-llama-server.ps1 - run llama-server for jev-local (replaces textgen's llama.cpp loader for this model)
#
# Prerequisites
#   * Official prebuilt llama.cpp for Windows + CUDA (llama-bXXXX-bin-win-cuda-x64.zip, plus the cudart zip
#     if you don't have the CUDA runtime installed). Use a build from 2026-09-23 or later.
#   * Your Qwen3.8-27B GGUF (the same file textgen loads).
#
# Edit the four settings below, then run:  .\start-llama-server.ps1
# Anything else that used textgen's API (SillyTavern, n8n, ...) can use this server's OpenAI-compatible
# endpoints at http://<host>:<port>/v1 instead.

$LlamaDir  = "C:\llama.cpp"                                   # folder containing llama-server.exe
$Model     = "C:\models\Qwen3.8-27B-Q5_K_M.gguf"              # the GGUF textgen loads
$CtxSize   = 131072                                            # textgen "ctx-size"; keep what you use there
$Port      = 5005                                              # jev-local's default LLAMA_URL is http://127.0.0.1:5005

$args = @(
    "--model", $Model,
    "--ctx-size", $CtxSize,
    "--gpu-layers", "999",             # textgen: gpu-layers (all layers on the A6000)
    "--flash-attn", "on",              # textgen always passes this
    "--batch-size", "2048",            # textgen defaults (batch-size / ubatch-size)
    "--ubatch-size", "512",
    "--port", $Port,
    "--host", "127.0.0.1",             # jev-local runs on this machine. Use 0.0.0.0 to expose the OpenAI API on the LAN.
    "--no-webui",
    "--parallel", "1",                 # ONE slot. jev-local relies on this slot's prompt cache; more slots re-read the document.

    # Hybrid-model checkpoints (Qwen3.8 is Gated-DeltaNet + attention). Each checkpoint stores the recurrent
    # state in host RAM (~100 MB for a 27B; the exact size is printed in the log with --verbose at
    # "created context checkpoint"). 64 keeps the end-of-document checkpoint alive through dependent forms
    # of up to ~30 fields; the default 32 covers ~15. Raise if you have RAM to spare and use longer forms.
    "--ctx-checkpoints", "64",

    # Speculative decoding via the MTP head that ships inside the Qwen3.8 GGUF (textgen "spec-type: draft-mtp").
    # Irrelevant to jev-local (every jev request decodes at most one token) but keeps ordinary chat fast.
    "--spec-type", "draft-mtp"

    # Optional:
    # "--api-key", "change-me",        # then set LLAMA_API_KEY for jev-local and the key in your other clients
    # "--threads", "8",
)

Write-Host "llama-server on http://127.0.0.1:$Port  (model: $Model, ctx: $CtxSize)"
& (Join-Path $LlamaDir "llama-server.exe") @args
