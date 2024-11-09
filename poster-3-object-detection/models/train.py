import torch

def train_model(model, dataloader, criterion_cls, criterion_bbox, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            images = images.cuda()
            targets_cls = torch.tensor([t['label'] for t in targets]).cuda()
            
            # Find potholes 
            fg_indices = [i for i, t in enumerate(targets) if t['label'] == 1]

            # Bounding box transformation targets for pothole proposals only
            fg_bbox_transforms = []
            for i in fg_indices:
                t = targets[i]
                
                # Use scaled ground truth bounding box values
                tx = (t['gt_bbox_xmin_scaled'] - t['image_xmin']) / (t['image_xmax'] - t['image_xmin'])
                ty = (t['gt_bbox_ymin_scaled'] - t['image_ymin']) / (t['image_ymax'] - t['image_ymin'])
                tw = torch.log((t['gt_bbox_xmax_scaled'] - t['gt_bbox_xmin_scaled']) / (t['image_xmax'] - t['image_xmin']))
                th = torch.log((t['gt_bbox_ymax_scaled'] - t['gt_bbox_ymin_scaled']) / (t['image_ymax'] - t['image_ymin']))
                
                fg_bbox_transforms.append(torch.tensor([tx, ty, tw, th]))

            fg_bbox_transforms = torch.stack(fg_bbox_transforms).cuda() if fg_bbox_transforms else torch.empty((0, 4)).cuda()

            # Forward pass
            optimizer.zero_grad()
            outputs_cls, outputs_bbox_transforms = model(images)
            
            # Classification Loss: Applies to both potholes and background
            loss_cls = criterion_cls(outputs_cls, targets_cls)

            # Regression Loss: Applies only to pothole proposals (i,.e not background)
            if fg_indices:
                outputs_bbox_fg = outputs_bbox_transforms[fg_indices]  
                loss_bbox = criterion_bbox(outputs_bbox_fg, fg_bbox_transforms)
            else:
                loss_bbox = torch.tensor(0.0).cuda()  # No regression loss for background 
            
            # Combine losses (TODO: figure out some weighting?)
            loss = loss_cls + loss_bbox

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')