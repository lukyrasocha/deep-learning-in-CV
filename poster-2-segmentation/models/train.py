import torch
from utils.logger import logger
import wandb

def train_model(model, train_loader, val_loader, loss_fn, optimizer, wandb_config, num_epochs=10, device='cuda'):

    wandb.init(
        project="project2-segmentation",
        config= wandb_config
    )

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

    wandb.finish() 
    logger.success("Training completed.")