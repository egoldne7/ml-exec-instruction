# GPT2
```bash
cd /tmp && rm -rf /tmp/ml-exec-instruction 
git clone https://github.com/egoldne7/ml-exec-instruction 
```
## In single node 8xH100
```bash
image_name="huggingface/transformers-pytorch-gpu:latest" # for CUDA >12.1
docker run \
--rm -it \
--cap-add CAP_SYS_PTRACE --shm-size="8g" \
--cap-add CAP_SYS_PTRACE --ipc host \
--gpus all \
--name fsdp_stresstest \
-v $(pwd)/ml-exec-instruction:/workspace \
$image_name \
 torchrun --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_id=1 \
    --rdzv_endpoint=127.0.0.1:29400 \
    --rdzv_backend=c10d \
    /workspace/benchmark/single_node_GPT2ForQA_fakedata.py --backend nccl --epochs 5000 --batch-size 80
```
## in multiple node 8xH100
```bash
cd /tmp && rm -rf /tmp/ml-exec-instruction 
git clone https://github.com/egoldne7/ml-exec-instruction 
NODE_ID=0 N_GPU_EACH_NODE=8 N_NODES=2 python3 /tmp/ml-exec-instruction/script/multinode.py
# ...
NODE_ID=1 N_GPU_EACH_NODE=8 N_NODES=2 python3 /tmp/ml-exec-instruction/script/multinode.py
```
