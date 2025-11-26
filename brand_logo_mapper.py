"""
Brand to Logo Image Mapping
Maps detected brands to actual logo images from OpenLogo dataset
"""

import os
import json
import re
import xml.etree.ElementTree as ET
from glob import glob
from collections import defaultdict


def clean_logo_name(raw_name: str) -> str:
    """
    Normalize OpenLogo class names to match brand keys
    
    Examples:
        'paypal_text' -> 'paypal'
        'ups_fig_1' -> 'ups'
        'amazon2' -> 'amazon'
    """
    # Remove suffixes
    cleaned = re.sub(r'(_text|_fig|_logo|\d+)$', '', raw_name. lower())
    # Remove underscores/hyphens
    cleaned = re.sub(r'[_\-\s]+', '', cleaned)
    return cleaned. strip()


def build_brand_to_images_index(jpeg_dir: str, anno_dir: str, 
                                 max_images_per_brand: int = 50):
    """
    Parse OpenLogo annotations and create brand -> image paths mapping
    
    Args:
        jpeg_dir: Path to JPEGImages folder
        anno_dir: Path to Annotations folder
        max_images_per_brand: Limit images per brand (for memory)
    
    Returns:
        Dict: {brand_key: [image_path1, image_path2, ...]}
    """
    print("Building brand-to-images index...")
    
    brand_to_images = defaultdict(list)
    xml_files = glob(os.path.join(anno_dir, "*.xml"))
    
    print(f"Found {len(xml_files)} annotation files")
    
    for idx, xml_path in enumerate(xml_files):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(xml_files)} files...")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image filename
            filename_node = root.find("filename")
            if filename_node is None:
                continue
            
            img_name = filename_node.text
            img_path = os.path.join(jpeg_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            # Get all logo brands in this image
            for obj in root.findall("object"):
                name_node = obj.find("name")
                if name_node is None:
                    continue
                
                logo_class = name_node.text. strip()
                brand_key = clean_logo_name(logo_class)
                
                if not brand_key:
                    continue
                
                # Limit images per brand
                if len(brand_to_images[brand_key]) < max_images_per_brand:
                    brand_to_images[brand_key].append(img_path)
        
        except Exception as e:
            # Skip corrupted XML files
            continue
    
    # Remove duplicates and convert to regular dict
    brand_map = {}
    for brand, paths in brand_to_images.items():
        unique_paths = list(set(paths))
        if len(unique_paths) > 0:
            brand_map[brand] = unique_paths
    
    print(f"\n✅ Built index for {len(brand_map)} brands")
    return brand_map


def save_brand_index(brand_map: dict, output_path: str):
    """Save brand index to JSON file"""
    with open(output_path, 'w') as f:
        json. dump(brand_map, f, indent=2)
    print(f"Saved brand index to {output_path}")


def load_brand_index(json_path: str) -> dict:
    """Load brand index from JSON file"""
    with open(json_path, 'r') as f:
        brand_map = json.load(f)
    print(f"Loaded brand index: {len(brand_map)} brands")
    return brand_map


# Build the index (run once)
if __name__ == "__main__":
    JPEG_DIR = "openlogo/JPEGImages"
    ANNO_DIR = "openlogo/Annotations"
    OUTPUT_JSON = "brand_to_images. json"
    
    print("="*60)
    print("Building Brand-to-Images Index")
    print("="*60)
    
    brand_map = build_brand_to_images_index(JPEG_DIR, ANNO_DIR)
    
    # Show sample
    print("\nSample brand mappings:")
    for brand, paths in list(brand_map.items())[:10]:
        print(f"  {brand}: {len(paths)} images")
    
    # Save
    save_brand_index(brand_map, OUTPUT_JSON)
    
    print("\n" + "="*60)
    print("Done!  Use brand_to_images.json in your dataset.")
    print("="*60)
