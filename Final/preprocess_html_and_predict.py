"""
HTML Email Preprocessing & Phishing Detection Script
Loads trained fusion model and analyzes .html email files

Usage:
    python preprocess_html_and_predict.py suspicious_email.html
"""

import sys
import os
import re
import json
import torch
import torch.nn as nn
from PIL import Image
from bs4 import BeautifulSoup
import torchvision.transforms as T
import base64
from io import BytesIO

# ============================================================================
# 1.MODEL ARCHITECTURE (Copy from notebook)
# ============================================================================

class TextFeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2, 2)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.relu_fc1 = nn.ReLU()
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.global_pool(x).squeeze(-1)
        x = self.relu_fc1(self.fc1(x))
        return x


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_features = nn.Linear(512, 512)
        self.relu_fc = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu_fc(self.fc_features(x))
        return x


class DualTowerFusionModel(nn.Module):
    def __init__(self, text_extractor, image_extractor, metadata_dim, num_classes, dropout):
        super().__init__()
        self.text_tower = text_extractor
        self.image_tower = image_extractor
        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim, 64), nn.ReLU(), nn.Dropout(0.25)
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(832, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, texts, metadata):
        text_features = self.text_tower(texts)
        image_features = self.image_tower(images)
        metadata_features = self.metadata_proj(metadata)
        combined = torch.cat([text_features, image_features, metadata_features], dim=1)
        logits = self.fusion_classifier(combined)
        return logits


# ============================================================================
# 2.HTML PARSING
# ============================================================================

def parse_html_email(html_path):
    """Extract subject, body text, and images from HTML email"""
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract subject
    subject = ""
    if soup.title:
        subject = soup.title.string
    elif soup.h1:
        subject = soup.h1.get_text(strip=True)
    else:
        first_text = soup.find(['p', 'div'])
        if first_text:
            subject = first_text.get_text(strip=True)[:100]
    
    # Extract body text
    for script in soup(["script", "style"]):
        script.decompose()
    body = soup.get_text(separator=' ', strip=True)
    
    # Extract images
    images = []
    for img_tag in soup.find_all('img'):
        img_src = img_tag.get('src', '')
        try:
            if img_src.startswith('data:image'):
                img_data = img_src.split(',')[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                images.append(img)
            elif img_src.startswith('http'):
                import requests
                response = requests.get(img_src, timeout=5)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                images.append(img)
            elif os.path.exists(img_src):
                img = Image.open(img_src).convert('RGB')
                images.append(img)
        except Exception as e:
            continue
    
    return {'subject': subject, 'body': body, 'images': images}


# ============================================================================
# 3.TEXT PREPROCESSING
# ============================================================================

def preprocess_text(subject, body, vocab_path='vocab_text_1.json', max_len=512):
    """Convert email text to model input tensor"""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        stoi = {w: i for i, w in enumerate(vocab_data['itos'])}
    
    # Tokenize
    full_text = f"{subject} {body}".lower()
    full_text = re.sub(r'[^a-z0-9\s]', ' ', full_text)
    tokens = full_text.split()
    
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    
    # Numericalize
    unk_idx = stoi.get('<unk>', 0)
    token_ids = [stoi.get(token, unk_idx) for token in tokens]
    
    # Pad
    if len(token_ids) < max_len:
        token_ids = token_ids + [0] * (max_len - len(token_ids))
    
    return torch.tensor(token_ids, dtype=torch.long)


# ============================================================================
# 4. IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(pil_image):
    """Convert PIL Image to model input tensor"""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_image)


# ============================================================================
# 5. METADATA EXTRACTION (Simplified)
# ============================================================================

def extract_metadata(subject, body):
    """Extract 20 metadata features"""
    text = f"{subject} {body}".lower()
    
    # URL features
    urls = re.findall(r'http[s]?://\S+', text)
    num_urls = min(len(urls), 10)
    has_shortened = float(any(short in url.lower() for url in urls for short in ['bit.ly', 'tinyurl', 'goo.gl']))
    has_suspicious_domain = float(any(susp in url.lower() for url in urls for susp in ['verify', 'secure', 'account', 'login']))
    
    # Keyword counts
    urgency_words = ['urgent', 'immediate', 'act now', 'expires', 'suspended', 'locked']
    num_urgency = sum(text.count(w) for w in urgency_words)
    
    action_words = ['verify', 'confirm', 'update', 'click here', 'validate']
    num_action = sum(text.count(w) for w in action_words)
    
    financial_words = ['account', 'payment', 'refund', 'money', 'prize']
    num_financial = sum(text.count(w) for w in financial_words)
    
    # Character features
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    num_exclaim = text.count('!')
    num_dollar = text.count('$')
    
    # Simplified 20-dim vector
    metadata = [
        len(text) / 1000.0,  # text length
        len(subject) / 100.0,  # subject length
        len(body) / 1000.0,  # body length
        num_urls,
        has_shortened,
        has_suspicious_domain,
        min(num_urgency, 10),
        min(num_action, 10),
        min(num_financial, 10),
        caps_ratio,
        min(num_exclaim, 10),
        min(num_dollar, 10),
        0.0,  # placeholder
        len(text.split()) / 100.0,  # word count
        0.0,  # generic greeting (simplified)
        0.0,  # CTA phrases (simplified)
        0.0,  # repeated chars (simplified)
        0.0,  # has IP (simplified)
        0.0,  # link mismatch (simplified)
        0.0   # suspicious TLD (simplified)
    ]
    
    return torch.tensor(metadata, dtype=torch.float)


# ============================================================================
# 6.MAIN PREDICTION FUNCTION
# ============================================================================

def predict_phishing(html_path, model, device='cpu'):
    """Complete pipeline: HTML → Prediction"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {html_path}")
    print(f"{'='*70}\n")
    
    # Parse HTML
    print("Step 1: Parsing HTML...")
    parsed = parse_html_email(html_path)
    print(f"   Subject: {parsed['subject'][:80]}...")
    print(f"   Body length: {len(parsed['body'])} chars")
    print(f"   Images found: {len(parsed['images'])}")
    
    # Preprocess text
    print("\nStep 2: Processing text...")
    text_tensor = preprocess_text(parsed['subject'], parsed['body'])
    print(f"   Text tensor shape: {text_tensor.shape}")
    
    # Preprocess image
    print("\nStep 3: Processing image...")
    if parsed['images']:
        image_tensor = preprocess_image(parsed['images'][0])
        print(f"   Using first image of size {parsed['images'][0].size}")
    else:
        image_tensor = torch.zeros(3, 224, 224)
        print("   No images found - using dummy tensor")
    
    # Extract metadata
    print("\nStep 4: Extracting metadata...")
    metadata_tensor = extract_metadata(parsed['subject'], parsed['body'])
    print(f"   Metadata tensor shape: {metadata_tensor.shape}")
    
    # Run model
    print("\nStep 5: Running fusion model...")
    model.eval()
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        text_batch = text_tensor.unsqueeze(0).to(device)
        metadata_batch = metadata_tensor.unsqueeze(0).to(device)
        
        outputs = model(image_batch, text_batch, metadata_batch)
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    
    # Generate report
    prediction = "PHISHING" if pred == 1 else "LEGITIMATE"
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    # Analyze metadata for suspicious signals
    metadata_vals = metadata_tensor.tolist()
    print(f"\nDetected Issues:")
    
    if metadata_vals[4] > 0.5:
        print(f"Contains shortened URLs (bit.ly, tinyurl)")
    if metadata_vals[5] > 0.5:
        print(f"Suspicious domain keywords detected")
    if metadata_vals[6] > 3:
        print(f"High urgency language ({int(metadata_vals[6])} urgent keywords)")
    if metadata_vals[7] > 2:
        print(f"Multiple call-to-action phrases")
    if metadata_vals[9] > 0.3:
        print(f"Excessive capitalization ({metadata_vals[9]*100:.1f}%)")
    if metadata_vals[10] > 3:
        print(f"{int(metadata_vals[10])} exclamation marks")
    
    print(f"\n{'='*70}\n")
    
    return {'prediction': prediction, 'confidence': confidence}


# ============================================================================
# 7.LOAD MODEL
# ============================================================================

def load_fusion_model(device='cpu'):
    """Load trained fusion model"""
    print("Loading fusion model...")
    
    with open('vocab_text_1.json', 'r') as f:
        vocab_data = json.load(f)
        vocab_size = len(vocab_data['itos'])
    
    text_extractor = TextFeatureExtractor(vocab_size, 128)
    image_extractor = ImageFeatureExtractor()
    
    model = DualTowerFusionModel(
        text_extractor, image_extractor, 20, 2, 0.5
    ).to(device)
    
    model.load_state_dict(torch.load('best_fusion_model.pth', map_location=device, weights_only=False))
    model.eval()
    
    print("Model loaded successfully\n")
    return model


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess_html_and_predict.py <email.html>")
        sys.exit(1)
    
    html_path = sys.argv[1]
    
    if not os.path.exists(html_path):
        print(f"Error: File not found: {html_path}")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_fusion_model(device)
    result = predict_phishing(html_path, model, device)
    
    # Save report
    report_path = html_path.replace('.html', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Phishing Detection Report\n")
        f.write(f"{'='*70}\n")
        f.write(f"File: {html_path}\n")
        f.write(f"Prediction: {result['prediction']}\n")
        f.write(f"Confidence: {result['confidence']*100:.2f}%\n")
    
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
