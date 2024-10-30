import torch
from utils.logger import logger
from models.metrics import dice_overlap, IoU, accuracy, sensitivity, specificity
import wandb

def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=10, device='cuda'):

    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # Train loop 
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation Loop
        model.eval()  
        val_dice = 0.0
        val_iou = 0.0
        val_accuracy = 0.0
        val_sensitivity = 0.0
        val_specificity = 0.0
        num_val_batches = 0
        val_loss = 0.0
        with torch.no_grad():  
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                
                val_loss += loss.item()

                num_val_batches += 1

                # Compute metrics for this batch
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                val_dice += dice_overlap(preds, masks)
                val_iou += IoU(preds, masks)
                val_accuracy += accuracy(preds, masks)
                val_sensitivity += sensitivity(preds, masks)
                val_specificity += specificity(preds, masks)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / num_val_batches
        avg_val_iou = val_iou / num_val_batches
        avg_val_accuracy = val_accuracy / num_val_batches
        avg_val_sensitivity = val_sensitivity / num_val_batches
        avg_val_specificity = val_specificity / num_val_batches

        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()  # Threshold predictions to 0 or 1

        wandb.log({"train_loss": avg_train_loss , "val_loss": avg_val_loss})
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "val_iou": avg_val_iou,
            "val_accuracy": avg_val_accuracy,
            "val_sensitivity": avg_val_sensitivity,
            "val_specificity": avg_val_specificity
        })

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    logger.success("Training completed.")

def train_model_weak(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # Train loop 
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            #print('Unique values in masks:', torch.unique(masks))
            
            optimizer.zero_grad()
            
            outputs = model(images)
            #print('outputs shape:', outputs.shape)
            #print(outputs)

            valid_mask = ~torch.isnan(masks)
            num_valid_pixels = valid_mask.sum().item()
            #print(f'Number of valid pixels in batch: {num_valid_pixels}')
            if num_valid_pixels == 0:
                print('No valid pixels in this batch. Skipping loss computation.')
                continue
            loss = loss_fn(outputs, masks)
            # add test print statement 
            #print('loss:', loss)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation Loop
        model.eval()  
        val_loss = 0.0
        with torch.no_grad():  
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"train_loss": avg_train_loss , "val_loss": avg_val_loss})

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    logger.success("Training completed.")