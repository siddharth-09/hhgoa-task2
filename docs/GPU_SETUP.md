# GPU ingest setup (RTX 3050, 4GB)

Only the **ingest** stage benefits. Serving stays CPU-only: retrieval is ~3ms on a
CPU, so a GPU at serve time would be spend with nothing to buy.

Expected: **3 hours → ~20-40 minutes** for the 10k x 2 run.

`multilingual-e5-small` is 118M params (~470MB fp32), so 4GB VRAM is comfortable —
batch 256 uses roughly 1.5-2GB.

---

## Step 1 — NVIDIA driver

**Linux**
```bash
nvidia-smi          # if this prints a table with a CUDA version, the driver is fine
```
If not:
```bash
sudo ubuntu-drivers autoinstall && sudo reboot     # Ubuntu/Pop!_OS
sudo apt install nvidia-driver-550 && sudo reboot  # Debian
```

**Windows** — install the GeForce driver from nvidia.com, then run everything inside
WSL2 (Ubuntu). WSL2 uses the *Windows* driver; do **not** install a Linux NVIDIA
driver inside WSL, it will break the passthrough.

You need driver **>= 525** for CUDA 12.x.

## Step 2 — Docker with GPU passthrough

```bash
# NVIDIA Container Toolkit (Ubuntu/Debian/WSL2)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify passthrough **before** building anything:
```bash
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```
If that does not print your 3050, stop and fix it here. Nothing downstream will work.

## Step 3 — Build and smoke-test

```bash
git clone <repo> && cd TASK2_IVR
docker build -f Dockerfile.gpu -t hhgoa-task2:gpu .
docker run --rm --gpus all hhgoa-task2:gpu
```

Expected:
```
providers: ['CUDAExecutionProvider', 'CPUExecutionProvider', ...]
variant=fp32 provider=CUDAExecutionProvider batch=256
```

**If it prints `CPUExecutionProvider`, stop.** The container asserts on this
deliberately: a CUDA/cuDNN mismatch does not raise, onnxruntime just drops the
provider and runs on CPU at ~1/20th the speed. You would get a "successful" run
that took three hours and told you nothing.

## Step 4 — Run the ingest

```bash
docker volume create task2-data
docker run --rm --gpus all -v task2-data:/data hhgoa-task2:gpu \
  python -m ingest.pipeline --langs hin mar --max-queries 10000 \
  --tag full --stream --variants fixed_256 semantic_128 metadata_128
```

Watch VRAM in another terminal:
```bash
watch -n2 nvidia-smi
```
If it OOMs, lower the batch: `-e EMBED_BATCH=128`.

## Step 5 — Move the index to the serving box

The index is a portable artifact — architecture-independent, so it does not matter
that it was built on CUDA and will be served on CPU.

```bash
docker run --rm -v task2-data:/data -v "$PWD:/out" hhgoa-task2:gpu \
  tar czf /out/index-full.tgz -C /data index/full raw

rsync -avz --partial --progress -e "ssh -i ~/.ssh/hhgoa-oracle" \
  index-full.tgz shared@80.225.231.132:~/
```

~2GB for 10k x 2, so 5-30 minutes depending on upload speed.

---

## Notes

**fp32, not int8, on GPU.** int8 is a *CPU* optimisation — its kernels target
AVX-VNNI or ARM dot-product instructions. On CUDA an int8 graph adds
dequantise/requantise work and can be slower than fp32. `default_variant()` handles
this automatically; only override `E5_VARIANT` if you are measuring.

**Batch 256, not 64.** On GPU the bottleneck moves from arithmetic to kernel-launch
overhead, and a 64-row batch leaves the device mostly idle.

**Only embedding is accelerated.** Streaming, chunking and index building stay on
the CPU, so expect ~20-40 min rather than a pure 10-30x on the wall clock.

**4GB is enough but not roomy.** Close other GPU consumers — a browser with hardware
acceleration can hold several hundred MB.
