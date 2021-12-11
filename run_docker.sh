docker run -it --rm \
        --network host \
        --ipc=host \
        --gpus all \
        -v /home/brianyao/Documents/intent2021icra:/workspace/intent2021ijcai \
        -v /mnt/workspace/users/brianyao/intent2021icra/checkpoints:/workspace/intent2021ijcai/checkpoints \
        -v /mnt/workspace/users/brianyao/intent2021icra/outputs:/workspace/intent2021ijcai/outputs \
        -v /mnt/workspace/users/brianyao/intent2021icra/wandb:/workspace/intent2021ijcai/wandb \
        -v /mnt/workspace/datasets:/workspace/intent2021ijcai/data \
        ped_pred:latest
