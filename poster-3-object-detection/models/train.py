import torch
from utils.logger import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import box_iou  # To compute IoU


def starts(n):
    print("*" * n)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torchvision.transforms as T




def train_model(model, train_loader, val_loader, criterion_cls, criterion_bbox, optimizer, num_epochs=1, iou_threshold=0.5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets, idx in train_loader:
            images = images.cuda()
            targets_cls = torch.tensor([t['label'] for t in targets]).cuda() # extract labels from targets in each batch:  0 for background 1 for pothole
            #print(targets_cls) # print the labels                           # example output: tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], device='cuda:0')
            #print("*"*50)
            #print(targets_cls)
            #print("*"*50)

            # Find potholes
            fg_indices = [i for i, t in enumerate(targets) if t['label'] == 1] # createst a list of indices of potholes in each batch
            #print("*"*50)
            #print(fg_indices)
            #print("*"*50)


            # Bounding box transformation targets for pothole proposals only
            fg_bbox_transforms = []
            for i in fg_indices:
                t = targets[i]
                # tx normalized x of of the gt box to the image bbox 
                tx = (t['gt_bbox_xmin'] - t['image_xmin']) / (t['image_xmax'] - t['image_xmin']) 
                # ty normalized y of of the gt box to the image bbox
                ty = (t['gt_bbox_ymin'] - t['image_ymin']) / (t['image_ymax'] - t['image_ymin']) 
                # logaritmic transformation of the width of the gt box to the width of the image
                tw = torch.log((t['gt_bbox_xmax'] - t['gt_bbox_xmin']) / (t['image_xmax'] - t['image_xmin']))
                # logaritmic transformation of the height of the gt box to the height of the image
                th = torch.log((t['gt_bbox_ymax'] - t['gt_bbox_ymin']) / (t['image_ymax'] - t['image_ymin']))
                fg_bbox_transforms.append(torch.tensor([tx, ty, tw, th]))

            fg_bbox_transforms = torch.stack(fg_bbox_transforms).cuda() if fg_bbox_transforms else torch.empty((0, 4)).cuda()

            # Forward pass
            optimizer.zero_grad()
            outputs_cls, outputs_bbox_transforms = model(images)

            # Classification Loss: Applies to both potholes and background
            loss_cls = criterion_cls(outputs_cls, targets_cls)

            # Regression Loss: Applies only to pothole proposals (i.e., not background)
            if fg_indices:
                outputs_bbox_fg = outputs_bbox_transforms[fg_indices]
                loss_bbox = criterion_bbox(outputs_bbox_fg, fg_bbox_transforms)
            else:
                loss_bbox = torch.tensor(0.0).cuda()  # No regression loss for background

            # Combine losses
            loss = loss_cls + loss_bbox

            # Backpropagation
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_loader)}')


        # Validation phase
        model.eval()
        #val_running_loss = 0.0
        val_cls_loss = 0.0
        val_bbox_loss = 0.0
        total_proposals = 0
        total_positive_proposals  = 0


        with torch.no_grad():
            for images, proposal_images_list, coords, image_ids, ground_truths in val_loader:
                # BATCH_SIZE = 1
                # images: [1, 3, H, W]
                proposals = coords[0]  # extracting proposal coordinates for the first image in the batch

                # GT boxes 
                gt_boxes = torch.stack([torch.tensor(
                    [gt['xmin'].item(), gt['ymin'].item(), 
                     gt['xmax'].item(), gt['ymax'].item()]) for gt in ground_truths[0]]).cuda() # Shape: [num_gt, 4]
                
                # Proposal boxes
                proposal_boxes = torch.stack(
                    [torch.tensor([t['xmin'].item(), t['ymin'].item(),
                                    t['xmax'].item(), t['ymax'].item()]) for t in proposals]).cuda() # Shape: [num_proposals, 4]
                
                # Compute IoU between proposals and ground truth boxes
                ious = box_iou(proposal_boxes, gt_boxes)  # Shape: [num_proposals, num_gt]
                max_ious, matched_gt_indices = ious.max(dim=1)  # Shape: [num_proposals]


                # Predict on all proposals (proposal_images_list[0] becuase we have only take one image in a batch)
                proposal_images = torch.stack(proposal_images_list[0]).cuda()
                 

                # Forward pass
                outputs_cls, outputs_bbox_transforms = model(proposal_images)


                predicted_classes = []

                # iterate over each proposal 
                for i in range(proposal_boxes.size(0)):
                    total_proposals += 1
                    pred_cls_logits = outputs_cls[i]  
                    pred_bbox = outputs_bbox_transforms[i]

                    # Apply softmax to get probabilities
                    cls_probs = torch.softmax(pred_cls_logits, dim=0)
                    pred_class = torch.argmax(cls_probs).item()  #outputs 1 or 0
                    predicted_classes.append(pred_class)

        

                    # Determine if there's a matching ground truth
                    iou = max_ious[i].item()
                    matched_gt_idx = matched_gt_indices[i].item() if iou >= iou_threshold else None
                    has_match = iou >= iou_threshold

                    # Classification Target
                    target_cls = 1 if has_match else 0

                    # Compute classification loss
                    loss_cls = criterion_cls(pred_cls_logits.unsqueeze(0), torch.tensor([target_cls]).cuda())
                    val_cls_loss += loss_cls.item()

                    # Compute if there's a positive match
                    if (pred_class == 1 and has_match) or (pred_class == 0 and has_match):
                        total_positive_proposals += 1

                        # Get the matched ground truth box
                        gt_box = gt_boxes[matched_gt_idx]

                        # Compute the transformation targets
                        tx = (gt_box[0] - proposal_boxes[i, 0]) / (proposal_boxes[i, 2] - proposal_boxes[i, 0])
                        ty = (gt_box[1] - proposal_boxes[i, 1]) / (proposal_boxes[i, 3] - proposal_boxes[i, 1])
                        tw = torch.log((gt_box[2] - gt_box[0]) / (proposal_boxes[i, 2] - proposal_boxes[i, 0]))
                        th = torch.log((gt_box[3] - proposal_boxes[i, 1]) / (proposal_boxes[i, 3] - proposal_boxes[i, 1]))

                        target_bbox = torch.tensor([tx, ty, tw, th]).cuda()

                        # Compute regression loss
                        loss_bbox = criterion_bbox(pred_bbox.unsqueeze(0), target_bbox.unsqueeze(0))
                        val_bbox_loss += loss_bbox.item()

                        # if i == 0: no need to calculate regression loss for background
                    else:
                        pass
                    
            # Calculate average losses
            avg_val_cls_loss = val_cls_loss / total_proposals if total_proposals > 0 else 0.0
            avg_val_bbox_loss = val_bbox_loss / total_positive_proposals if total_positive_proposals > 0 else 0.0

            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Classification Loss: {avg_val_cls_loss:.4f}, Validation Regression Loss: {avg_val_bbox_loss:.4f}')

