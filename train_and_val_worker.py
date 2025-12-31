"""TrainAndEvalWorker: Handles training, validation, and evaluation.

This module is responsible for:
- Creating DataLoader instances
- Instantiating models
- Training loop with optimization
- Validation loop
- Computing and tracking metrics
- Saving and loading model weights
- Final evaluation on test set

NEVER loads raw data or creates splits.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from typing import Dict, Optional, Tuple
import os
from utils.utils import (
    AverageMeter,
    calculate_dice_score,
    calculate_iou_score,
    get_learning_rate
)


class TrainAndEvalWorker:
    """Worker class for training, validation, and evaluation."""
    
    def __init__(self, config):
        """Initialize TrainAndEvalWorker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def _create_loss_function(self):
        """Create loss function based on config.
        
        Returns:
            Loss function
        """
        # Get class weights if enabled
        class_weights = None
        if self.config.use_class_weights:
            class_weights = torch.tensor(
                self.config.class_weights, 
                dtype=torch.float32
            ).to(self.device)
            print(f"Using class weights: {self.config.class_weights}")
        
        if self.config.loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        elif self.config.loss_type == "dice":
            return smp.losses.DiceLoss(mode='multilabel', from_logits=True)
        elif self.config.loss_type == "focal":
            return smp.losses.FocalLoss(
                mode='multilabel',
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma
            )
        elif self.config.loss_type == "dice_focal":
            # Combine Dice and Focal Loss with class weighting
            dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
            focal_loss = smp.losses.FocalLoss(
                mode='multilabel',
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma
            )
            
            class CombinedLoss(nn.Module):
                def __init__(self, dice_loss, focal_loss, dice_weight, focal_weight, class_weights=None):
                    super().__init__()
                    self.dice_loss = dice_loss
                    self.focal_loss = focal_loss
                    self.dice_weight = dice_weight
                    self.focal_weight = focal_weight
                    self.class_weights = class_weights
                
                def forward(self, pred, target):
                    dice = self.dice_loss(pred, target)
                    focal = self.focal_loss(pred, target)
                    
                    # Apply class weights if provided
                    if self.class_weights is not None:
                        # Compute per-class losses
                        batch_size = pred.size(0)
                        num_classes = pred.size(1)
                        
                        # Weighted combination
                        weighted_dice = 0
                        weighted_focal = 0
                        for c in range(num_classes):
                            # Extract class-specific predictions and targets
                            pred_c = pred[:, c:c+1, :, :]
                            target_c = target[:, c:c+1, :, :]
                            
                            dice_c = self.dice_loss(pred_c, target_c)
                            focal_c = self.focal_loss(pred_c, target_c)
                            
                            weighted_dice += self.class_weights[c] * dice_c
                            weighted_focal += self.class_weights[c] * focal_c
                        
                        # Normalize by sum of weights
                        dice = weighted_dice / self.class_weights.sum()
                        focal = weighted_focal / self.class_weights.sum()
                    
                    return self.dice_weight * dice + self.focal_weight * focal
            
            return CombinedLoss(
                dice_loss, focal_loss,
                self.config.dice_weight, self.config.focal_weight,
                class_weights
            )
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
    def create_model(self) -> nn.Module:
        """Create segmentation model.
        
        Returns:
            PyTorch model
        """
        if self.config.model_name == "unet":
            model = smp.Unet(
                encoder_name=self.config.encoder_name,
                encoder_weights=self.config.encoder_weights,
                in_channels=3,
                classes=self.config.num_classes,
                activation=None  # We'll apply sigmoid in forward
            )
        elif self.config.model_name == "unetplusplus":
            model = smp.UnetPlusPlus(
                encoder_name=self.config.encoder_name,
                encoder_weights=self.config.encoder_weights,
                in_channels=3,
                classes=self.config.num_classes,
                activation=None
            )
        elif self.config.model_name == "deeplabv3plus":
            model = smp.DeepLabV3Plus(
                encoder_name=self.config.encoder_name,
                encoder_weights=self.config.encoder_weights,
                in_channels=3,
                classes=self.config.num_classes,
                activation=None
            )
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")
        
        return model.to(self.device)
    
    def create_dataloader(
        self,
        dataset,
        shuffle: bool = True,
        batch_size: Optional[int] = None
    ) -> DataLoader:
        """Create DataLoader from dataset.
        
        Args:
            dataset: PyTorch Dataset
            shuffle: Whether to shuffle data
            batch_size: Batch size (uses config if None)
            
        Returns:
            DataLoader
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def train(
        self,
        train_dataset,
        val_dataset,
        fold: Optional[int] = None
    ) -> Tuple[nn.Module, Dict]:
        """Train model on given datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            fold: Optional fold number for logging
            
        Returns:
            Tuple of (trained_model, history_dict)
        """
        # Create dataloaders
        train_loader = self.create_dataloader(train_dataset, shuffle=True)
        val_loader = self.create_dataloader(val_dataset, shuffle=False)
        
        # Create model
        model = self.create_model()
        
        # Loss function
        criterion = self._create_loss_function()
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        if self.config.scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.max_learning_rate,
                epochs=self.config.num_epochs,
                steps_per_epoch=len(train_loader) // self.config.accumulation_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': []
        }
        
        best_dice = 0.0
        epochs_without_improvement = 0
        best_model_path = os.path.join(
            self.config.checkpoint_dir,
            f"best_model_fold{fold}.pth" if fold is not None else "best_model.pth"
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, scheduler
            )
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_dice, val_iou = self._validate_epoch(
                model, val_loader, criterion
            )
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            history['val_iou'].append(val_iou)
            
            # Learning rate scheduling (for ReduceLROnPlateau only)
            if self.config.scheduler_type == "plateau":
                scheduler.step(val_dice)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            print(f"LR: {get_learning_rate(optimizer):.6f}")
            
            # Save best model and early stopping
            if val_dice > best_dice:
                best_dice = val_dice
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': best_dice,
                    'config': self.config
                }, best_model_path)
                print(f"Saved best model with Dice: {best_dice:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs")
                    print(f"Best Dice: {best_dice:.4f}")
                    break
        
        # Load best model
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, history
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler = None
    ) -> float:
        """Train for one epoch with gradient accumulation.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (for OneCycleLR)
            
        Returns:
            Average training loss
        """
        model.train()
        loss_meter = AverageMeter()
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Normalize loss for accumulation
            loss = loss / self.config.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Optimizer step every accumulation_steps
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Step scheduler if OneCycleLR
                if scheduler is not None and self.config.scheduler_type == "onecycle":
                    scheduler.step()
            
            # Update metrics (use original loss scale)
            loss_meter.update(loss.item() * self.config.accumulation_steps, images.size(0))
            pbar.set_postfix({'loss': loss_meter.avg})
        
        # Final optimizer step if there are remaining gradients
        if (batch_idx + 1) % self.config.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        return loss_meter.avg
    
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float, float]:
        """Validate for one epoch.
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (avg_loss, avg_dice, avg_iou)
        """
        model.eval()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        iou_meter = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Apply sigmoid for metrics
                preds = torch.sigmoid(outputs)
                
                # Calculate metrics
                dice = calculate_dice_score(preds, masks).mean()
                iou = calculate_iou_score(preds, masks).mean()
                
                # Update meters
                loss_meter.update(loss.item(), images.size(0))
                dice_meter.update(dice.item(), images.size(0))
                iou_meter.update(iou.item(), images.size(0))
                
                pbar.set_postfix({
                    'loss': loss_meter.avg,
                    'dice': dice_meter.avg,
                    'iou': iou_meter.avg
                })
        
        return loss_meter.avg, dice_meter.avg, iou_meter.avg
    
    def evaluate(
        self,
        test_dataset,
        model_path: str
    ) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            test_dataset: Test dataset
            model_path: Path to saved model weights
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating model from {model_path}")
        
        # Create model and load weights
        model = self.create_model()
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create dataloader
        test_loader = self.create_dataloader(test_dataset, shuffle=False)
        
        # Metrics
        dice_meter = AverageMeter()
        iou_meter = AverageMeter()
        dice_per_class = [AverageMeter() for _ in range(self.config.num_classes)]
        iou_per_class = [AverageMeter() for _ in range(self.config.num_classes)]
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                
                # Calculate metrics
                dice = calculate_dice_score(preds, masks)
                iou = calculate_iou_score(preds, masks)
                
                # Update overall metrics
                dice_meter.update(dice.mean().item(), images.size(0))
                iou_meter.update(iou.mean().item(), images.size(0))
                
                # Update per-class metrics
                for class_idx in range(self.config.num_classes):
                    dice_per_class[class_idx].update(
                        dice[class_idx].item(), images.size(0)
                    )
                    iou_per_class[class_idx].update(
                        iou[class_idx].item(), images.size(0)
                    )
                
                pbar.set_postfix({
                    'dice': dice_meter.avg,
                    'iou': iou_meter.avg
                })
        
        # Compile results
        results = {
            'mean_dice': dice_meter.avg,
            'mean_iou': iou_meter.avg
        }
        
        for class_idx, class_name in enumerate(self.config.classes):
            results[f'dice_{class_name}'] = dice_per_class[class_idx].avg
            results[f'iou_{class_name}'] = iou_per_class[class_idx].avg
        
        # Print results
        print("\n=== Test Results ===")
        print(f"Mean Dice: {results['mean_dice']:.4f}")
        print(f"Mean IoU: {results['mean_iou']:.4f}")
        for class_name in self.config.classes:
            print(f"{class_name.capitalize()} - Dice: {results[f'dice_{class_name}']:.4f}, "
                  f"IoU: {results[f'iou_{class_name}']:.4f}")
        
        return results
    
    def _apply_tta_transform(self, image: torch.Tensor, transform_name: str) -> torch.Tensor:
        """Apply TTA transformation to image.
        
        Args:
            image: Input image tensor [B, C, H, W]
            transform_name: Name of transformation
            
        Returns:
            Transformed image
        """
        if transform_name == 'original':
            return image
        elif transform_name == 'horizontal_flip':
            return torch.flip(image, dims=[3])
        elif transform_name == 'vertical_flip':
            return torch.flip(image, dims=[2])
        elif transform_name == 'rotate_90':
            return torch.rot90(image, k=1, dims=[2, 3])
        elif transform_name == 'rotate_180':
            return torch.rot90(image, k=2, dims=[2, 3])
        elif transform_name == 'rotate_270':
            return torch.rot90(image, k=3, dims=[2, 3])
        else:
            return image
    
    def _reverse_tta_transform(self, pred: torch.Tensor, transform_name: str) -> torch.Tensor:
        """Reverse TTA transformation on prediction.
        
        Args:
            pred: Prediction tensor [B, C, H, W]
            transform_name: Name of transformation to reverse
            
        Returns:
            Reversed prediction
        """
        if transform_name == 'original':
            return pred
        elif transform_name == 'horizontal_flip':
            return torch.flip(pred, dims=[3])
        elif transform_name == 'vertical_flip':
            return torch.flip(pred, dims=[2])
        elif transform_name == 'rotate_90':
            return torch.rot90(pred, k=3, dims=[2, 3])  # Reverse is k=3
        elif transform_name == 'rotate_180':
            return torch.rot90(pred, k=2, dims=[2, 3])
        elif transform_name == 'rotate_270':
            return torch.rot90(pred, k=1, dims=[2, 3])  # Reverse is k=1
        else:
            return pred
    
    def evaluate_with_tta(
        self,
        test_dataset,
        model_path: str
    ) -> Dict[str, float]:
        """Evaluate model on test set with Test Time Augmentation.
        
        Args:
            test_dataset: Test dataset
            model_path: Path to saved model weights
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating with TTA from {model_path}")
        print(f"TTA Transforms: {self.config.tta_transforms}")
        
        # Create model and load weights
        model = self.create_model()
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create dataloader
        test_loader = self.create_dataloader(test_dataset, shuffle=False)
        
        # Metrics
        dice_meter = AverageMeter()
        iou_meter = AverageMeter()
        dice_per_class = [AverageMeter() for _ in range(self.config.num_classes)]
        iou_per_class = [AverageMeter() for _ in range(self.config.num_classes)]
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing with TTA")
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Accumulate predictions from all TTA transforms
                tta_preds = []
                for transform_name in self.config.tta_transforms:
                    # Apply transform
                    transformed_img = self._apply_tta_transform(images, transform_name)
                    
                    # Forward pass
                    outputs = model(transformed_img)
                    preds = torch.sigmoid(outputs)
                    
                    # Reverse transform on prediction
                    preds = self._reverse_tta_transform(preds, transform_name)
                    tta_preds.append(preds)
                
                # Average all TTA predictions
                final_preds = torch.stack(tta_preds).mean(dim=0)
                
                # Calculate metrics
                dice = calculate_dice_score(final_preds, masks)
                iou = calculate_iou_score(final_preds, masks)
                
                # Update overall metrics
                dice_meter.update(dice.mean().item(), images.size(0))
                iou_meter.update(iou.mean().item(), images.size(0))
                
                # Update per-class metrics
                for class_idx in range(self.config.num_classes):
                    dice_per_class[class_idx].update(
                        dice[class_idx].item(), images.size(0)
                    )
                    iou_per_class[class_idx].update(
                        iou[class_idx].item(), images.size(0)
                    )
                
                pbar.set_postfix({
                    'dice': dice_meter.avg,
                    'iou': iou_meter.avg
                })
        
        # Compile results
        results = {
            'mean_dice': dice_meter.avg,
            'mean_iou': iou_meter.avg
        }
        
        for class_idx, class_name in enumerate(self.config.classes):
            results[f'dice_{class_name}'] = dice_per_class[class_idx].avg
            results[f'iou_{class_name}'] = iou_per_class[class_idx].avg
        
        # Print results
        print("\n=== Test Results (with TTA) ===")
        print(f"Mean Dice: {results['mean_dice']:.4f}")
        print(f"Mean IoU: {results['mean_iou']:.4f}")
        for class_name in self.config.classes:
            print(f"{class_name.capitalize()} - Dice: {results[f'dice_{class_name}']:.4f}, "
                  f"IoU: {results[f'iou_{class_name}']:.4f}")
        
        return results
    
    def evaluate_ensemble(
        self,
        test_dataset,
        model_paths: list,
        use_tta: bool = False
    ) -> Dict[str, float]:
        """Evaluate using ensemble of multiple models.
        
        Args:
            test_dataset: Test dataset
            model_paths: List of paths to saved model weights
            use_tta: Whether to use TTA for each model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating ensemble of {len(model_paths)} models")
        if use_tta:
            print(f"With TTA: {self.config.tta_transforms}")
        
        # Load all models
        models = []
        for path in model_paths:
            model = self.create_model()
            checkpoint = torch.load(path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        
        # Create dataloader
        test_loader = self.create_dataloader(test_dataset, shuffle=False)
        
        # Metrics
        dice_meter = AverageMeter()
        iou_meter = AverageMeter()
        dice_per_class = [AverageMeter() for _ in range(self.config.num_classes)]
        iou_per_class = [AverageMeter() for _ in range(self.config.num_classes)]
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing with Ensemble")
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Accumulate predictions from all models
                ensemble_preds = []
                
                for model in models:
                    if use_tta:
                        # TTA for this model
                        tta_preds = []
                        for transform_name in self.config.tta_transforms:
                            transformed_img = self._apply_tta_transform(images, transform_name)
                            outputs = model(transformed_img)
                            preds = torch.sigmoid(outputs)
                            preds = self._reverse_tta_transform(preds, transform_name)
                            tta_preds.append(preds)
                        # Average TTA predictions for this model
                        model_pred = torch.stack(tta_preds).mean(dim=0)
                    else:
                        # Single prediction
                        outputs = model(images)
                        model_pred = torch.sigmoid(outputs)
                    
                    ensemble_preds.append(model_pred)
                
                # Average all model predictions
                final_preds = torch.stack(ensemble_preds).mean(dim=0)
                
                # Calculate metrics
                dice = calculate_dice_score(final_preds, masks)
                iou = calculate_iou_score(final_preds, masks)
                
                # Update overall metrics
                dice_meter.update(dice.mean().item(), images.size(0))
                iou_meter.update(iou.mean().item(), images.size(0))
                
                # Update per-class metrics
                for class_idx in range(self.config.num_classes):
                    dice_per_class[class_idx].update(
                        dice[class_idx].item(), images.size(0)
                    )
                    iou_per_class[class_idx].update(
                        iou[class_idx].item(), images.size(0)
                    )
                
                pbar.set_postfix({
                    'dice': dice_meter.avg,
                    'iou': iou_meter.avg
                })
        
        # Compile results
        results = {
            'mean_dice': dice_meter.avg,
            'mean_iou': iou_meter.avg
        }
        
        for class_idx, class_name in enumerate(self.config.classes):
            results[f'dice_{class_name}'] = dice_per_class[class_idx].avg
            results[f'iou_{class_name}'] = iou_per_class[class_idx].avg
        
        # Print results
        tta_str = " + TTA" if use_tta else ""
        print(f"\n=== Test Results (Ensemble{tta_str}) ===")
        print(f"Mean Dice: {results['mean_dice']:.4f}")
        print(f"Mean IoU: {results['mean_iou']:.4f}")
        for class_name in self.config.classes:
            print(f"{class_name.capitalize()} - Dice: {results[f'dice_{class_name}']:.4f}, "
                  f"IoU: {results[f'iou_{class_name}']:.4f}")
        
        return results
