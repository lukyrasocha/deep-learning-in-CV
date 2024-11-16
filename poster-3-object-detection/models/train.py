import torch
from utils.logger import logger
from torchvision.ops import box_iou

def train_model(
    model, train_loader, val_loader, criterion_cls, criterion_bbox,
    optimizer, num_epochs=1, iou_threshold=0.5, cls_weight=1, reg_weight=1
):
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
            cls_running_loss += loss_cls.item()

            # Regression Loss
            if fg_indices:
                outputs_bbox_fg = outputs_bbox_transforms[fg_indices]
                loss_bbox = criterion_bbox(outputs_bbox_fg, fg_bbox_transforms)
            else:
                loss_bbox = torch.tensor(0.0).cuda()

            bbox_running_loss += loss_bbox.item()

            # Combine Losses
            loss = cls_weight * loss_cls + reg_weight * loss_bbox
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_cls_loss = cls_running_loss / len(train_loader)
        avg_train_bbox_loss = bbox_running_loss / len(train_loader)
        avg_train_loss = running_loss / len(train_loader)

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

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f} (Cls: {avg_train_cls_loss:.4f}, Reg: {avg_train_bbox_loss:.4f}) - "
            f"Val Loss: {avg_val_loss:.4f} (Cls: {avg_val_cls_loss:.4f}, Reg: {avg_val_bbox_loss:.4f})"
        )