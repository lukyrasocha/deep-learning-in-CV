import datetime
import numpy as np
import torch
from utils.logger import logger
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import os
from utils.metrics import non_max_suppression
from utils.metrics import calculate_precision_recall, calculate_mAP, non_max_suppression
from torchvision.transforms import ToTensor
import wandb


# for debugging
def starts(n):
    print("*"*n)

def train_model(
    model, train_loader, val_loader, criterion_cls, criterion_bbox,
    optimizer, num_epochs=1, iou_threshold=0.5, cls_weight=1, reg_weight=1, 
    experiment_name="experiment", patience=10, min_delta=1e-4
):
    
    wandb.init(
        project="object_detection",  # Set your W&B project name
        name=experiment_name,        # Name of the experiment
        config={                     # Log hyperparameters
            "num_epochs": num_epochs,
            "iou_threshold": iou_threshold,
            "cls_weight": cls_weight,
            "reg_weight": reg_weight,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
    )    

    train_losses = []
    val_losses = []

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0


    for epoch in range(num_epochs):
        # ------------------------
        # Training Phase
        # ------------------------
        model.train()
        running_loss = 0.0
        cls_running_loss = 0.0
        bbox_running_loss = 0.0

        for images, targets, idx in train_loader:
            images = images.cuda()
            targets_cls = torch.tensor([t['label'] for t in targets]).cuda()

            # Find positive proposals
            fg_indices = [i for i, t in enumerate(targets) if t['label'] == 1]

            # Compute bounding box transforms for positive proposals
            fg_bbox_transforms = []
            for i in fg_indices:
                t = targets[i]
                tx = (t['gt_bbox_xmin'] - t['image_xmin']) / (t['image_xmax'] - t['image_xmin'])
                ty = (t['gt_bbox_ymin'] - t['image_ymin']) / (t['image_ymax'] - t['image_ymin'])
                tw = torch.log((t['gt_bbox_xmax'] - t['gt_bbox_xmin']) / (t['image_xmax'] - t['image_xmin']))
                th = torch.log((t['gt_bbox_ymax'] - t['gt_bbox_ymin']) / (t['image_ymax'] - t['image_ymin']))
                fg_bbox_transforms.append(torch.tensor([tx, ty, tw, th]))

            fg_bbox_transforms = torch.stack(fg_bbox_transforms).cuda() if fg_bbox_transforms else torch.empty((0, 4)).cuda()


            # Forward pass
            optimizer.zero_grad()
            outputs_cls, outputs_bbox_transforms = model(images)

            # Classification Loss
            loss_cls = criterion_cls(outputs_cls, targets_cls)
            cls_running_loss += cls_weight * loss_cls.item()

            # Regression Loss
            if fg_indices:
                outputs_bbox_fg = outputs_bbox_transforms[fg_indices]
                loss_bbox = criterion_bbox(outputs_bbox_fg, fg_bbox_transforms)
            else:
                loss_bbox = torch.tensor(0.0).cuda()

            bbox_running_loss += reg_weight * loss_bbox.item()
            #print(f"Targets_cls: {targets_cls}")
            #print(f"Number of positive samples: {(targets_cls == 1).sum()}")

            # Combine Losses
            loss = cls_weight * loss_cls + reg_weight * loss_bbox
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_cls_loss = cls_running_loss / len(train_loader)
        avg_train_bbox_loss = bbox_running_loss / len(train_loader)
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

                # Log training metrics to W&B
        wandb.log({
            "train/cls_loss": avg_train_cls_loss,
            "train/bbox_loss": avg_train_bbox_loss,
            "train/total_loss": avg_train_loss,
            "epoch": epoch + 1
        })
        


        # ------------------------
        # Validation Phase
        # ------------------------
        model.eval()
        val_cls_loss = 0.0
        val_bbox_loss = 0.0
        total_proposals = 0
        total_positive_proposals = 0

        with torch.no_grad():
            for images, proposal_images_list, coords, image_ids, ground_truths in val_loader:
                proposals = coords[0]

                gt_boxes = torch.stack([torch.tensor([gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax']]) for gt in ground_truths[0]]).cuda()
                proposal_boxes = torch.stack([torch.tensor([t['xmin'], t['ymin'], t['xmax'], t['ymax']]) for t in proposals]).cuda()

                # Compute IoU
                ious = box_iou(proposal_boxes, gt_boxes)
                max_ious, matched_gt_indices = ious.max(dim=1)

                # Process proposals
                proposal_images = torch.stack(proposal_images_list[0]).cuda() # num_proposals x 3 x 256 x 256
                outputs_cls, outputs_bbox_transforms = model(proposal_images)

                assert outputs_cls.shape[1] == 2, "Must be two, (Logit for background and for pothole)"

                for i in range(proposal_boxes.size(0)):
                    total_proposals += 1
                    pred_cls_logits = outputs_cls[i]
                    pred_bbox = outputs_bbox_transforms[i]

                    cls_probs = torch.softmax(pred_cls_logits, dim=0)
                    pred_class = torch.argmax(cls_probs).item()

                    iou = max_ious[i].item()
                    matched_gt_idx = matched_gt_indices[i].item() if iou >= iou_threshold else None
                    has_match = iou >= iou_threshold
                    target_cls = 1 if has_match else 0

                    loss_cls = criterion_cls(pred_cls_logits.unsqueeze(0), torch.tensor([target_cls]).cuda())
                    val_cls_loss += cls_weight * loss_cls.item()

                    if has_match:
                        total_positive_proposals += 1
                        gt_box = gt_boxes[matched_gt_idx]
                        tx = (gt_box[0] - proposal_boxes[i, 0]) / (proposal_boxes[i, 2] - proposal_boxes[i, 0])
                        ty = (gt_box[1] - proposal_boxes[i, 1]) / (proposal_boxes[i, 3] - proposal_boxes[i, 1])
                        tw = torch.log((gt_box[2] - gt_box[0]) / (proposal_boxes[i, 2] - proposal_boxes[i, 0]))
                        th = torch.log((gt_box[3] - gt_box[1]) / (proposal_boxes[i, 3] - proposal_boxes[i, 1]))
                        target_bbox = torch.tensor([tx, ty, tw, th]).cuda()

                        loss_bbox = criterion_bbox(pred_bbox.unsqueeze(0), target_bbox.unsqueeze(0))
                        val_bbox_loss += reg_weight * loss_bbox.item()

        avg_val_cls_loss = val_cls_loss / total_proposals if total_proposals > 0 else 0.0
        avg_val_bbox_loss = val_bbox_loss / total_positive_proposals if total_positive_proposals > 0 else 0.0
        avg_val_loss = avg_val_cls_loss + avg_val_bbox_loss
        val_losses.append(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset counter if validation loss improves
        else:
            patience_counter += 1

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f} (Cls: {avg_train_cls_loss:.4f}, Reg: {avg_train_bbox_loss:.4f}) - "
            f"Val Loss: {avg_val_loss:.4f} (Cls: {avg_val_cls_loss:.4f}, Reg: {avg_val_bbox_loss:.4f})"
        )

                # Log validation metrics to W&B
        wandb.log({
            "val/cls_loss": avg_val_cls_loss,
            "val/bbox_loss": avg_val_bbox_loss,
            "val/total_loss": avg_val_loss
        })

        # Stop training if patience is exceeded
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    wandb.finish()


        # ------------------------
    # Plot and Save Loss Curves
    # ----------------------------
    # Save Precision-Recall curve with unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    figures_dir_png = f"figures/png/RCNN_train_val_loss_{experiment_name}_{timestamp}"
    figures_dir_svg = f"figures/svg/RCNN_train_val_loss_{experiment_name}_{timestamp}"
    os.makedirs(figures_dir_png, exist_ok=True)
    os.makedirs(figures_dir_svg, exist_ok=True)

    # Dynamically determine the number of completed epochs
    completed_epochs = len(train_losses)

    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(range(1, completed_epochs + 1), train_losses, label='RCNN Train Loss', color='#990000', marker='o')

    # Plot validation loss
    plt.plot(range(1, completed_epochs + 1), val_losses, label='RCNN Val Loss', color='#2F3EEA', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss', fontsize=18, color='#990000')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    # Save plots
    plt.savefig(os.path.join(figures_dir_png, "loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir_svg, "loss_curve.svg"), format='svg', bbox_inches='tight')


def evaluate_model(model, val_loader, split="val", iou_threshold=0.5, confidence_threshold=0.8, experiment_name="experiment"):
    # Set model to evaluation mode

    # Define custom colors
    color_primary = '#990000'  # University red
    color_secondary = '#2F3EEA'  # University blue
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Lists to store per-image ground truths and predictions
    ground_truths = []  # List[List[Dict]]
    predictions = []    # List[List[Dict]]

    with torch.no_grad():
        for images, proposal_images_list, coords, image_ids, ground_truths_batch in val_loader:
            batch_size = len(image_ids)
            # For each image in the batch
            for idx in range(batch_size):
                image_id = image_ids[idx]
                # Process ground truths for this image
                gt_boxes = []
                for gt in ground_truths_batch[idx]:
                    xmin = gt.get('xmin').item()
                    ymin = gt.get('ymin').item()
                    xmax = gt.get('xmax').item()
                    ymax = gt.get('ymax').item()
                    gt_boxes.append({
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    })
                ground_truths.append(gt_boxes)
                
                # Get proposals for the image
                proposal_images = torch.stack(proposal_images_list[idx]).to(device)

                outputs_cls, outputs_bbox_transforms = model(proposal_images)

                # Convert outputs to CPU numpy arrays
                outputs_cls = outputs_cls.detach().cpu()
                outputs_bbox_transforms = outputs_bbox_transforms.detach().cpu()

                # Process outputs 
                scores = torch.softmax(outputs_cls, dim=1)[:, 1]  # Get pothole scores
                boxes = coords[idx]  # coords for this image
                # After computing scores in evaluate_model
                #print(f"Scores before thresholding: {scores}")


                # Filter out low-confidence proposals
                mask = scores >= confidence_threshold
                scores = scores[mask]
                boxes = [boxes[i] for i in range(len(boxes)) if mask[i]]
                #print(f"Number of scores before thresholding: {len(scores)}")

                if len(scores) == 0:
                    predictions.append([])  # No predictions for this image
                    continue

                # Prepare predictions for this image
                pred_boxes = []
                for i in range(len(scores)):
                    pred_boxes.append({
                        'pre_class': float(scores[i]),
                        'pre_bbox_xmin': float(boxes[i]['xmin']),
                        'pre_bbox_ymin': float(boxes[i]['ymin']),
                        'pre_bbox_xmax': float(boxes[i]['xmax']),
                        'pre_bbox_ymax': float(boxes[i]['ymax']),
                    })
                # Apply NMS using your provided function
                nms_results = non_max_suppression(pred_boxes, iou_threshold=iou_threshold)

                predictions.append(nms_results)

    # After processing all batches
    # Now calculate precision and recall
    precision_values, recall_values = calculate_precision_recall(ground_truths, predictions, iou_threshold)

    # Calculate mAP
    mAP = calculate_mAP(precision_values, recall_values)

    # Convert precision and recall lists to numpy arrays for plotting
    precision = np.array(precision_values)
    recall = np.array(recall_values)

    # Save Precision-Recall curve with unique filename

    os.makedirs(f'figures/png/{experiment_name}', exist_ok=True)
    os.makedirs(f'figures/svg/{experiment_name}', exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pr_curve_filename_png = f"figures/png/{experiment_name}/precision_recall_curve_{split}_{experiment_name}_{timestamp}.png"
    pr_curve_filename_svg = f"figures/svg/{experiment_name}/precision_recall_curve_{split}_{experiment_name}_{timestamp}.svg"
    # Ensure directories exist
    os.makedirs("figures/png", exist_ok=True)
    os.makedirs("figures/svg", exist_ok=True)

       # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        recall, precision, 
        color=color_primary, 
        marker='o', 
        linestyle='-', 
        linewidth=2, 
        markersize=5, 
        label=r'\textbf{Precision-Recall Curve}'
    )
    plt.xlabel('Recall', fontsize=16, labelpad=10, color='black')  # Black text for labels
    plt.ylabel('Precision', fontsize=16, labelpad=10, color='black')  # Black text for labels
    plt.title(r'\textbf{Precision-Recall Curve}', fontsize=18, pad=15, color=color_primary)
    plt.legend(loc='best', fontsize=16, title_fontsize=18, frameon=True)  # Larger legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
    plt.tight_layout()

    # Save figures
    plt.savefig(pr_curve_filename_png, dpi=300, bbox_inches='tight')
    plt.savefig(pr_curve_filename_svg, format='svg', bbox_inches='tight')
    plt.close() 

    logger.info(f"Precision-Recall curve saved as {pr_curve_filename_png} and {pr_curve_filename_svg}")

    return mAP, precision, recall
