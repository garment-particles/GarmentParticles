from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata

def check_checkpoint_compatibility(checkpoint_path, model, app_name="app.model"):
    """Compare checkpoint keys with model keys."""
    # Get checkpoint metadata
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata.keys())
    # only keep the keys that are in the app_name
    checkpoint_keys = {k for k in checkpoint_keys if k.startswith(app_name)}
    
    # Get current model keys (with the "app.model." prefix to match checkpoint)
    model_keys = set(f"{app_name}.{k}" for k in model.state_dict().keys())
    
    missing_in_ckpt = model_keys - checkpoint_keys
    unexpected_in_ckpt = checkpoint_keys - model_keys
    common_keys = model_keys & checkpoint_keys
    
    return missing_in_ckpt, unexpected_in_ckpt, common_keys