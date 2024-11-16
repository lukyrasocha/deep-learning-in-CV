import torch
from utils.logger import logger

def train_model(model, train_loader, val_loader, criterion_cls, criterion_bbox, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets, idx in train_loader:
            images = images.cuda()
            targets_cls = torch.tensor([t['label'] for t in targets]).cuda()

            # Find potholes
            fg_indices = [i for i, t in enumerate(targets) if t['label'] == 1]

            # Bounding box transformation targets for pothole proposals only
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
        val_running_loss = 0.0
        with torch.no_grad():
            for images, proposal_images_list, coords, image_ids, ground_truths in val_loader:
                images = torch.stack(proposal_images_list[0]).cuda()
                targets_cls = torch.tensor([gt['labels'] for gt in ground_truths[0]]).cuda()  # Directly using ground truth labels

                print(len(coords[0]))
                print(len(ground_truths[0]))

                print(ground_truths[0][0])
                print(coords[0][0])
                # Find pothole proposals
                fg_indices = [i for i, t in enumerate(coords[0]) if ground_truths[0][i]['labels'] == 1]

                # Bounding box transformation targets for pothole proposals only
                fg_bbox_transforms = []
                for i in fg_indices:
                    t = coords[0][i]
                    tx = (t['gt_bbox_xmin'] - t['image_xmin']) / (t['image_xmax'] - t['image_xmin'])
                    ty = (t['gt_bbox_ymin'] - t['image_ymin']) / (t['image_ymax'] - t['image_ymin'])
                    tw = torch.log((t['gt_bbox_xmax'] - t['gt_bbox_xmin']) / (t['image_xmax'] - t['image_xmin']))
                    th = torch.log((t['gt_bbox_ymax'] - t['gt_bbox_ymin']) / (t['image_ymax'] - t['image_ymin']))
                    fg_bbox_transforms.append(torch.tensor([tx, ty, tw, th]))

                fg_bbox_transforms = torch.stack(fg_bbox_transforms).cuda() if fg_bbox_transforms else torch.empty((0, 4)).cuda()

                # Forward pass
                outputs_cls, outputs_bbox_transforms = model(images)

                # Classification Loss
                loss_cls = criterion_cls(outputs_cls, targets_cls)

                # Regression Loss
                if fg_indices:
                    outputs_bbox_fg = outputs_bbox_transforms[fg_indices]
                    loss_bbox = criterion_bbox(outputs_bbox_fg, fg_bbox_transforms)
                else:
                    loss_bbox = torch.tensor(0.0).cuda()

                # Total Validation Loss
                val_loss = loss_cls + loss_bbox
                val_running_loss += val_loss.item()
            
        # Validation phase
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_running_loss / len(val_loader)}')