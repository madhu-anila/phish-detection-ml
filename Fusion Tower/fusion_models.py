"""
Multi-Modal Fusion Architecture with Metadata Integration
UPDATED: Now includes metadata feature vector input
Author: Srihari-Narayan
Date: 2025-01-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextFeatureExtractor(nn.Module):
    """
    Feature extractor from trained CustomCNN_Text
    Outputs 256-dim features (matching your trained model)
    """
    def __init__(self, vocab_size, embed_dim=128, pretrained_state_dict=None):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Block 1: embed_dim -> 64 channels
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 2: 64 -> 128 channels
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self. pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 3: 128 -> 256 channels
        self.conv3 = nn. Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 4: 256 -> 512 channels
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature projection to 256-dim
        self.fc1 = nn.Linear(512, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        
        # Load pretrained weights if provided
        if pretrained_state_dict is not None:
            self._load_pretrained(pretrained_state_dict)
        
        self.feature_dim = 256
    
    def _load_pretrained(self, state_dict):
        """Load weights from trained text CNN"""
        own_state = self.state_dict()
        loaded_count = 0
        
        for name, param in state_dict.items():
            # Skip final classification layers
            if name.startswith('fc2') or name.startswith('fc3') or name.startswith('dropout'):
                continue
            
            if name in own_state:
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded_count += 1
                else:
                    print(f"  Skipping {name}: shape mismatch")
        
        print(f" Loaded {loaded_count} pretrained text CNN layers")
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] - token IDs
        Returns:
            features: [batch_size, 256] - text feature vector
        """
        # Embedding: [B, T] -> [B, T, E]
        x = self.embedding(x)
        
        # Transpose for Conv1d: [B, T, E] -> [B, E, T]
        x = x.transpose(1, 2)
        
        # Conv blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self. conv2(x))))
        x = self.pool3(self. relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self. relu4(self.bn4(self.conv4(x))))
        
        # Global pooling: [B, 512, T] -> [B, 512, 1] -> [B, 512]
        x = self.global_pool(x). squeeze(-1)
        
        # Feature projection: [B, 512] -> [B, 256]
        x = self.relu_fc1(self.fc1(x))
        
        return x


class ImageFeatureExtractor(nn.Module):
    """
    Feature extractor from trained CustomCNN_Logo
    Outputs 512-dim features
    """
    def __init__(self, pretrained_state_dict=None):
        super().__init__()
        
        # Conv Block 1: 3 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Conv Block 2: 64 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Conv Block 3: 128 -> 256
        self. conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Conv Block 4: 256 -> 512
        self. conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection (keep 512-dim)
        self.fc_features = nn.Linear(512, 512)
        self.relu_fc = nn.ReLU(inplace=True)
        
        # Load pretrained weights if provided
        if pretrained_state_dict is not None:
            self._load_pretrained(pretrained_state_dict)
        
        self.feature_dim = 512
    
    def _load_pretrained(self, state_dict):
        """Load weights from trained image CNN"""
        own_state = self.state_dict()
        loaded_count = 0
        
        for name, param in state_dict.items():
            # Skip final classifier layers
            if name.startswith('classifier'):
                # Only load first FC layer as fc_features
                if 'classifier. 0' in name:
                    new_name = name.replace('classifier.0', 'fc_features')
                    if new_name in own_state and own_state[new_name]. shape == param.shape:
                        own_state[new_name].copy_(param)
                        loaded_count += 1
                continue
            
            if name in own_state:
                if own_state[name].shape == param. shape:
                    own_state[name].copy_(param)
                    loaded_count += 1
                else:
                    print(f" Skipping {name}: shape mismatch")
        
        print(f"Loaded {loaded_count} pretrained image CNN layers")
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, 224, 224] - images
        Returns:
            features: [batch_size, 512] - image feature vector
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu_fc(self.fc_features(x))
        return x


class DualTowerFusionModel(nn.Module):
    """
    Dual-Tower Multi-Modal Fusion with Metadata
    Combines: Text (256) + Image (512) + Metadata (20) = 788-dim
    """
    def __init__(self, text_extractor, image_extractor, 
                 metadata_dim=20, num_classes=2, dropout=0.5):
        super().__init__()
        
        self.text_tower = text_extractor
        self.image_tower = image_extractor
        
        # Metadata projection
        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Calculate combined feature dimension
        # Text: 256, Image: 512, Metadata: 64 -> Total: 832
        combined_dim = (self.text_tower.feature_dim + 
                       self.image_tower. feature_dim + 64)
        
        # Fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn. Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn. Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn. Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn. Dropout(dropout),
            
            nn.Linear(128, num_classes)
        )
        
        # Initialize fusion layers
        self._initialize_fusion_weights()
    
    def _initialize_fusion_weights(self):
        """Initialize fusion classifier weights"""
        for m in self.fusion_classifier. modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn. init.constant_(m.bias, 0)
        
        # Initialize metadata projection
        for m in self.metadata_proj. modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images, texts, metadata):
        """
        Args:
            images: [batch_size, 3, 224, 224] - logo images
            texts: [batch_size, seq_len] - email text token IDs
            metadata: [batch_size, 20] - metadata feature vector
        
        Returns:
            logits: [batch_size, num_classes] - classification scores
        """
        # Extract features from all three modalities
        text_features = self.text_tower(texts)          # [batch, 256]
        image_features = self.image_tower(images)       # [batch, 512]
        metadata_features = self. metadata_proj(metadata) # [batch, 64]
        
        # Concatenate all features
        combined = torch.cat([text_features, image_features, metadata_features], dim=1)
        # [batch, 832]
        
        # Fusion classification
        logits = self.fusion_classifier(combined)  # [batch, num_classes]
        
        return logits
    
    def freeze_towers(self):
        """Freeze both feature extractors - only train fusion layers"""
        for param in self. text_tower.parameters():
            param.requires_grad = False
        for param in self.image_tower.parameters():
            param. requires_grad = False
        print(" Frozen text and image towers (only fusion + metadata trainable)")
    
    def unfreeze_towers(self):
        """Unfreeze towers for end-to-end fine-tuning"""
        for param in self.text_tower.parameters():
            param.requires_grad = True
        for param in self.image_tower.parameters():
            param.requires_grad = True
        print(" Unfrozen all towers (end-to-end training enabled)")
    
    def get_trainable_params(self):
        """Return count of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Print detailed model information"""
        total = sum(p.numel() for p in self.parameters())
        trainable = self.get_trainable_params()
        frozen = total - trainable
        
        print("\n" + "="*60)
        print("Model Architecture Summary")
        print("="*60)
        print(f"Text Tower:     {self.text_tower. feature_dim}-dim output")
        print(f"Image Tower:    {self.image_tower.feature_dim}-dim output")
        print(f"Metadata:       64-dim (projected from 20)")
        print(f"Combined:       832-dim")
        print(f"\nTotal params:      {total:,}")
        print(f"Trainable params:  {trainable:,}")
        print(f"Frozen params:     {frozen:,}")
        print(f"Model size:        ~{total * 4 / 1024 / 1024:.2f} MB")
        print("="*60 + "\n")
