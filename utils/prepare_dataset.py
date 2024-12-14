import fiftyone as fo
import fiftyone.zoo as foz
import requests
import os
from urllib.parse import unquote

def get_wikimedia_paintings():
    # Van Gogh paintings from Wikimedia Commons
    paintings = [
        "https://upload.wikimedia.org/wikipedia/commons/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/4c/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/76/Vincent_van_Gogh_-_De_slaapkamer_-_Google_Art_Project.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/5/5c/Van_Gogh_-_Terrasse_des_Caf%C3%A9s_an_der_Place_du_Forum_in_Arles_am_Abend1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/9/9d/Vincent_van_Gogh_-_Sunflowers_-_VGM_F458.jpg"
        # Add more URLs as needed
    ]
    return paintings

def prepare_dataset():
    # Download a small subset of COCO for content images
    content_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=100,
        shuffle=True
    )
    
    # Download paintings from Wikimedia
    style_dir = "data/style"
    os.makedirs(style_dir, exist_ok=True)
    
    # Define headers for Wikimedia API
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; YourApp/1.0; your@email.com)'  # Replace with your info
    }
    
    # Download paintings
    paintings = get_wikimedia_paintings()
    downloaded_files = []
    for i, url in enumerate(paintings):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            filename = f"{style_dir}/vangogh_{i}.jpg"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded painting {i+1}/{len(paintings)}")
            downloaded_files.append(filename)
        except Exception as e:
            print(f"Error downloading painting {i+1}: {str(e)}")
    
    if not downloaded_files:
        raise ValueError("No style images were downloaded successfully!")
    
    # Create FiftyOne dataset from downloaded images
    style_dataset = fo.Dataset.from_dir(
        dataset_dir=style_dir,
        dataset_type=fo.types.ImageDirectory,
        name="style_images"
    )
    
    if len(style_dataset) == 0:
        raise ValueError("Style dataset is empty after creation!")
    
    # Export the datasets to disk
    content_dataset.export(
        export_dir="data/photos",
        dataset_type=fo.types.ImageDirectory
    )
    
    print("Content dataset location:", content_dataset.first().filepath)
    print(f"Style dataset created with {len(style_dataset)} images")
    print("Style dataset location:", style_dataset.first().filepath)
    
    # Launch the FiftyOne App to visualize the datasets
    session = fo.launch_app(content_dataset)
    input("Press Enter to close the FiftyOne App...")
    session.close()

if __name__ == "__main__":
    print(foz.list_zoo_datasets())
    prepare_dataset()
    
