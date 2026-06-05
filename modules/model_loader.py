import torch


def load_actor_critic_checkpoint(actor_critic, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "actor_critic_state_dict" in checkpoint:
            state_dict = checkpoint["actor_critic_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    actor_critic.load_state_dict(state_dict, strict=False)
    actor_critic.to(device)
    actor_critic.eval()
    return actor_critic