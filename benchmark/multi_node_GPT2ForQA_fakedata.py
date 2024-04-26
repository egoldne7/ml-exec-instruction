from __future__ import print_function
import os

# from tensorboardX import SummaryWriter
# from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from transformers import GPT2Config, GPT2ForQuestionAnswering
# from torch.distributed.fsdpFSDP import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 2))
print("WORLD_SIZE: ", WORLD_SIZE)

def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    print("Starting main")
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI, dist.Backend.UCC],
                            default=dist.Backend.UCC)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')
    
    rank = int(os.environ.get('RANK', 0))

    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(args.backend, rank=rank, world_size=WORLD_SIZE)

    print("starting create model")
    configuration = GPT2Config()
    # Initializing a model from the bert-base-uncased style configuration
    model = GPT2ForQuestionAnswering(configuration)
    print(f'rank is: {rank}')
    # torch.cuda.set_device(rank)
    # torch.cuda.set to all gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # device = torch.device(device)
    model = model.to(device)
    print(f'Using DDP on cuda device: {next(model.parameters()).device}')
    # model = FSDP(model)
    model = DDP(model)

    batch_size = args.batch_size
    print(f"starting create data batch_size={batch_size}")

    sequence_length = 512
    # Create random input IDs (tokens)
    input_ids = torch.randint(0, configuration.vocab_size, (batch_size, sequence_length))
    # Create random attention masks
    attention_mask = torch.randint(0, 2, (batch_size, sequence_length))
    # Create random start and end positions for the answers
    start_positions = torch.randint(0, sequence_length, (batch_size,))
    end_positions = torch.randint(0, sequence_length, (batch_size,))

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    start_positions = start_positions.to(device)
    end_positions = end_positions.to(device)
    # Define a simple cross entropy loss
    loss_function = torch.nn.CrossEntropyLoss()
    # Define an optimizer
    # optimizer = Adam(model.parameters(), lr=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Training loop
    print("starting training")
    for epoch in range(args.epochs):  # Number of epochs
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    # loop job in tqdm

if __name__ == '__main__':
    main()