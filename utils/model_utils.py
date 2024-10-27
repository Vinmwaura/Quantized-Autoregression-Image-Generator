import os

import torch

# Save model checkpoints.
def save_model(
        dest_path,
        file_name,
        model_dict,
        logging=print):
    try:
        checkpoint_folder_path = os.path.join(
            dest_path,
            "models_checkpoint")

        # Create folder if doesn't exist.
        os.makedirs(checkpoint_folder_path, exist_ok=True)

        # File name for model to be saved.
        model_file_path = os.path.join(
            checkpoint_folder_path,
            file_name)

        # TODO: Move to using a safer save function not using pickle.
        torch.save(
            model_dict,
            model_file_path)
        success = True
    except Exception as e:
        logging(f"Exception occured while saving model: {e}.")
        success = False
    finally:
        return success

# Load model checkpoints.
def load_model(
        checkpoint_path,
        logging=print):
    # Check if file path exists.
    file_exists = os.path.exists(checkpoint_path)

    model_checkpoint = None
    if file_exists:
        # Load checkpoint param in CPU.
        model_checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device('cpu'))
        success = True
    else:
        success = False
        logging(f"Checkpoint does not exist.")
    return success, model_checkpoint
