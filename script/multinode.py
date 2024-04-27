import os
import subprocess

image_name="egoldne7/pytorch_ucc_transformers:latest"

head_node_ip=os.environ.get('head_node_ip', "103.77.130.16")
N_GPU_EACH_NODE=int(os.environ.get('N_GPU_EACH_NODE', 8))
N_NODES=int(os.environ.get('N_NODES', 2))
NODE_ID=int(os.environ.get('NODE_ID', 0))
EPOCHS=int(os.environ.get('EPOCHS', 5000))
WORLD_SIZE=N_GPU_EACH_NODE*N_NODES
print(f"head_node_ip: {head_node_ip}, N_GPU_EACH_NODE: {N_GPU_EACH_NODE}, N_NODES: {N_NODES}, NODE_ID: {NODE_ID}, WORLD_SIZE: {WORLD_SIZE}")

def exec_cmd(cmd):
   print(cmd)
   return os.system(cmd)

# check docker container if exists
cmd=f'docker ps -a | grep ddp_stresstest'
result = exec_cmd(cmd)
if result == 0:
   print("ddp_stresstest container exists, stop and remove it")    
   cmd='docker stop ddp_stresstest'
   exec_cmd(cmd)
   cmd='docker rm ddp_stresstest'
   exec_cmd(cmd)

cmd = f"""
docker run \
   --rm \
   --ulimit memlock=-1 \
   --network host \
   --cap-add CAP_SYS_PTRACE --shm-size="8g" \
   --cap-add CAP_SYS_PTRACE --ipc host \
   --gpus '"device=all"' \
   --device=/dev/infiniband \
   --volume /tmp/ml-exec-instruction:/workspace \
   --name ddp_stresstest \
   --detach \
   --env MASTER_ADDR={head_node_ip} \
   --env MASTER_PORT=1234 \
   --env WORLD_SIZE={WORLD_SIZE} \
   {image_name} \
   tail -f /dev/null
"""
exec_cmd(cmd)

for local_rank in range(N_GPU_EACH_NODE):
   global_rank = local_rank + NODE_ID * N_GPU_EACH_NODE
   cmd_format = f"RANK={global_rank} CUDA_VISIBLE_DEVICES={local_rank} python3 /workspace/benchmark/multi_node_GPT2ForQA_fakedata.py --backend ucc --epochs {EPOCHS} --batch-size 80"
   cmd = f'docker exec -d -it ddp_stresstest bash -c "{cmd_format}"'
   exec_cmd(cmd)