"""
ONE-TIME SETUP SCRIPT
Run this once to build the brand-to-images index
"""

from brand_logo_mapper import build_brand_to_images_index, save_brand_index

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ONE-TIME SETUP: Building Brand-to-Images Index")
    print("="*70)
    print("\nThis will take 2-5 minutes...")
    print("You only need to run this ONCE.\n")
    
    # Paths
    JPEG_DIR = "openlogo/JPEGImages"
    ANNO_DIR = "openlogo/Annotations"
    OUTPUT = "brand_to_images.json"
    
    # Build index
    brand_map = build_brand_to_images_index(JPEG_DIR, ANNO_DIR, max_images_per_brand=50)
    
    # Statistics
    total_images = sum(len(paths) for paths in brand_map.values())
    avg_per_brand = total_images / len(brand_map) if brand_map else 0
    
    print("\n" + "="*70)
    print("Index Statistics:")
    print("="*70)
    print(f"Total brands: {len(brand_map)}")
    print(f"Total image mappings: {total_images}")
    print(f"Average images per brand: {avg_per_brand:.1f}")
    
    # Show top brands by image count
    sorted_brands = sorted(brand_map.items(), key=lambda x: len(x[1]), reverse=True)
    print("\nTop 10 brands by image count:")
    for brand, paths in sorted_brands[:10]:
        print(f"  {brand}: {len(paths)} images")
    
    # Save
    save_brand_index(brand_map, OUTPUT)
    
    print("\n" + "="*70)
    print(f" Setup complete! Generated: {OUTPUT}")
    print("You can now run the fusion training.")
    print("="*70 + "\n")
