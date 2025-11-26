"""
UPDATED Multimodal Dataset with Brand-Based Image Pairing
Replaces fusion_dataset. py
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import random
import re
import json

from brand_extractor import extract_brands_from_text
from metadata_engineer import extract_metadata_features


def simple_tokenize(text):
    """Basic tokenizer (same as before)"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9@.\-_/ ]+", " ", text)
    tokens = text.split()
    return tokens


def numericalize(tokens, stoi_map, unk_idx=1):
    """Convert tokens to indices"""
    return [stoi_map. get(tok, unk_idx) for tok in tokens]


class BrandAwareMultiModalDataset(Dataset):
    """
    Multi-modal dataset with brand-based image pairing and metadata
    """
    def __init__(self, email_df, brand_to_images, stoi_map, 
                 image_transform=None, max_seq_length=512):
        """
        Args:
            email_df: DataFrame with ['subject', 'body', 'label']
            brand_to_images: Dict {brand_key: [img_path1, img_path2, ... ]}
            stoi_map: Token vocabulary dict
            image_transform: torchvision transforms
            max_seq_length: Max token length
        """
        self. emails = email_df. reset_index(drop=True)
        self.brand_to_images = brand_to_images
        self.stoi_map = stoi_map
        self.image_transform = image_transform
        self.max_seq_length = max_seq_length
        
        # Get all available images as fallback
        self.all_images = []
        for paths in brand_to_images.values():
            self.all_images.extend(paths)
        self.all_images = list(set(self.all_images))  # unique
        
        print(f"Dataset: {len(self.emails)} emails, "
              f"{len(brand_to_images)} brands, "
              f"{len(self. all_images)} unique logo images")
    
    def __len__(self):
        return len(self.emails)
    
    def __getitem__(self, idx):
        row = self.emails.iloc[idx]
        subject = str(row. get('subject', ''))
        body = str(row.get('body', ''))
        label = int(row['label'])
        
        # === TEXT PROCESSING ===
        full_text = subject + " " + body
        tokens = simple_tokenize(full_text)
        
        # Truncate
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        token_ids = numericalize(tokens, self.stoi_map)
        text_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        # === BRAND DETECTION & IMAGE PAIRING ===
        detected_brands = extract_brands_from_text(full_text)
        
        # Find image based on detected brands
        img_path = None
        if detected_brands:
            # Try to find a brand that has images
            for brand in detected_brands:
                if brand in self.brand_to_images:
                    img_path = random.choice(self.brand_to_images[brand])
                    break
        
        # Fallback to random image if no brand match
        if img_path is None and self.all_images:
            img_path = random.choice(self.all_images)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
        except Exception as e:
            # Fallback: create blank image if loading fails
            image = torch.zeros(3, 224, 224)
        
        # === METADATA EXTRACTION ===
        metadata = extract_metadata_features(subject, body)
        metadata_tensor = torch.tensor(metadata, dtype=torch.float)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'text': text_tensor,
            'image': image,
            'metadata': metadata_tensor,
            'label': label_tensor,
            'detected_brands': detected_brands  # For debugging
        }


def collate_multimodal_batch(batch):
    """
    Custom collate function for variable-length sequences
    """
    # Pad text sequences
    texts = [item['text'] for item in batch]
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Stack metadata
    metadata = torch.stack([item['metadata'] for item in batch])
    
    # Stack labels
    labels = torch.stack([item['label'] for item in batch])
    
    return images, padded_texts, metadata, labels


# Test dataset
if __name__ == "__main__":
    import pandas as pd
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    
    # Load data
    email_df = pd.read_csv("cleaned_combined_emails.csv")[['subject', 'body', 'label']].head(100)
    
    with open("brand_to_images. json", 'r') as f:
        brand_map = json.load(f)
    
    with open("vocab_text_1.json", 'r') as f:
        vocab_data = json.load(f)
        itos = vocab_data['itos']
        stoi = {w: i for i, w in enumerate(itos)}
    
    # Create transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = BrandAwareMultiModalDataset(
        email_df=email_df,
        brand_to_images=brand_map,
        stoi_map=stoi,
        image_transform=transform,
        max_seq_length=512
    )
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_multimodal_batch)
    
    print("\n" + "="*60)
    print("Testing Brand-Aware Dataset")
    print("="*60)
    
    for images, texts, metadata, labels in loader:
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Texts: {texts.shape}")
        print(f"  Metadata: {metadata.shape}")
        print(f"  Labels: {labels. shape}")
        
        # Show detected brands
        for i in range(min(2, len(dataset))):
            sample = dataset[i]
            print(f"\n  Sample {i}: Detected brands: {sample['detected_brands']}")
        
        break
    
    print("\n" + "="*60)
