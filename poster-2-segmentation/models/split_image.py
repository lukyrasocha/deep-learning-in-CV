import torch

def split_image_into_patches(input_image, patch_size, model):
    device = input_image.device  # Get the device of the input image

    patch = input_image[:, 0:patch_size, 0:patch_size]
    mask_pred = model(patch.unsqueeze(0)).squeeze(0)

    _, mask_height, mask_width = mask_pred.shape
    _, image_height, image_width = input_image.shape
    padding = int((patch_size - mask_height) / 2)

    # Initialize the mask on the correct device
    mask = torch.zeros(1, image_height, image_width, device=device)  

    i = 0
    while i < image_height:
        j = 0
        while j < image_width:

            if i + patch_size > image_height:
                start_i = image_height - patch_size 
                save_i = i + patch_size - image_height 
                delta_i = image_height - i - 2*padding
            else:
                start_i = i
                save_i = 0
                delta_i = patch_size - 2*padding

            end_i = start_i + patch_size

            if j + patch_size > image_width:
                start_j = image_width - patch_size
                save_j = j + patch_size - image_width 
                delta_j = image_width - j - 2*padding
            else:
                start_j = j
                save_j = 0
                delta_j = patch_size - 2*padding

            end_j = start_j + patch_size

            patch = input_image[:, start_i:end_i, start_j:end_j]
            
            # Ensure the patch is on the correct device
            patch = patch.to(device)

            # Process the patch with the model
            processed_patch = model(patch.unsqueeze(0)).squeeze(0)

            # Update the mask with the processed patch
            mask[:, i + padding: i + padding + delta_i, j + padding : j + padding + delta_j] = \
                processed_patch[:, save_i : patch_size - 2*padding, save_j : patch_size - 2*padding]

            j += patch_size - 2 * padding

        i += patch_size - 2 * padding

    mask = mask[:, padding : image_height - padding, padding : image_width - padding ]

    preds = torch.sigmoid(mask)
    predictions = (preds > 0.5).float()

    # Calculate padding
    pad_height = (input_image.shape[1] - predictions.shape[1]) // 2
    pad_width = (input_image.shape[2] - predictions.shape[2]) // 2

    padded_predictions = torch.nn.functional.pad(
        predictions,
        (pad_width, pad_width, pad_height, pad_height),
        mode="constant",
        value=0
    )

    return padded_predictions