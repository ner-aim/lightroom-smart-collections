"""
Synthetic Lightroom Catalog Data Generator

Generates realistic photo catalog data mimicking actual Lightroom Classic user patterns.
Creates 5,000+ photos with EXIF metadata, organizational data, and usage patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import random
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class LightroomCatalogGenerator:
    """Generates synthetic Lightroom catalog data with realistic patterns."""
    
    def __init__(self, num_photos: int = 5000):
        self.num_photos = num_photos
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime(2024, 11, 1)
        
        # Equipment definitions
        self.cameras = [
            "Canon EOS R5", "Canon EOS R6", 
            "Nikon Z9", "Nikon Z6 II",
            "Sony A7 IV", "Sony A7R V",
            "Fujifilm X-T5", "Fujifilm X-H2S"
        ]
        
        self.lenses = {
            "portrait": ["50mm f/1.8", "85mm f/1.4", "35mm f/1.4", "50mm f/1.2"],
            "landscape": ["16-35mm f/4", "24-70mm f/2.8", "14mm f/2.8"],
            "telephoto": ["70-200mm f/2.8", "100-400mm f/4.5-5.6"],
            "versatile": ["24-70mm f/2.8", "24-105mm f/4", "24-120mm f/4"]
        }
        
        self.file_extensions = {
            "Canon EOS R5": "CR3", "Canon EOS R6": "CR3",
            "Nikon Z9": "NEF", "Nikon Z6 II": "NEF",
            "Sony A7 IV": "ARW", "Sony A7R V": "ARW",
            "Fujifilm X-T5": "RAF", "Fujifilm X-H2S": "RAF"
        }
        
        # Genre keywords
        self.keywords_by_genre = {
            "portrait": ["portrait", "people", "headshot", "model", "studio", "natural light", "bokeh"],
            "wedding": ["wedding", "bride", "groom", "ceremony", "reception", "couple", "family"],
            "landscape": ["landscape", "nature", "mountain", "sunset", "sunrise", "scenic", "travel"],
            "event": ["event", "party", "corporate", "conference", "celebration", "candid"],
            "street": ["street", "urban", "city", "architecture", "candid", "documentary"],
            "nature": ["wildlife", "nature", "animals", "birds", "outdoors", "macro"]
        }
        
        self.locations = [
            "New York, NY", "San Francisco, CA", "Seattle, WA", 
            "Austin, TX", "Denver, CO", "Portland, OR",
            "Chicago, IL", "Boston, MA", None  # None for non-geotagged
        ]
        
    def _generate_photo_shoots(self) -> List[Dict]:
        """Generate realistic photo shoot sessions."""
        shoots = []
        current_date = self.start_date
        
        # Define shoot types with their characteristics
        shoot_types = {
            "wedding": {
                "duration_hours": 8,
                "photo_count": (200, 500),
                "genres": ["wedding"],
                "preferred_lenses": ["portrait", "versatile"],
                "time_of_day": ["morning", "afternoon", "golden_hour"]
            },
            "portrait_session": {
                "duration_hours": 2,
                "photo_count": (50, 150),
                "genres": ["portrait"],
                "preferred_lenses": ["portrait"],
                "time_of_day": ["golden_hour", "afternoon"]
            },
            "landscape": {
                "duration_hours": 3,
                "photo_count": (30, 100),
                "genres": ["landscape"],
                "preferred_lenses": ["landscape"],
                "time_of_day": ["golden_hour", "blue_hour", "sunrise"]
            },
            "event": {
                "duration_hours": 4,
                "photo_count": (100, 300),
                "genres": ["event"],
                "preferred_lenses": ["versatile", "portrait"],
                "time_of_day": ["afternoon", "night"]
            },
            "street": {
                "duration_hours": 3,
                "photo_count": (40, 120),
                "genres": ["street"],
                "preferred_lenses": ["versatile"],
                "time_of_day": ["afternoon", "golden_hour"]
            }
        }
        
        shoot_id = 0
        while current_date < self.end_date:
            # Random gap between shoots (1-14 days)
            gap_days = int(np.random.choice([1, 2, 3, 5, 7, 14], p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1]))
            current_date += timedelta(days=gap_days)
            
            if current_date >= self.end_date:
                break
            
            # Select shoot type with realistic distribution
            shoot_type = np.random.choice(
                list(shoot_types.keys()),
                p=[0.15, 0.35, 0.20, 0.20, 0.10]  # portrait sessions most common
            )
            
            shoot_config = shoot_types[shoot_type]
            num_photos = random.randint(*shoot_config["photo_count"])
            
            shoots.append({
                "shoot_id": shoot_id,
                "shoot_type": shoot_type,
                "date": current_date,
                "photo_count": num_photos,
                "duration_hours": shoot_config["duration_hours"],
                "genres": shoot_config["genres"],
                "preferred_lenses": shoot_config["preferred_lenses"],
                "time_of_day": random.choice(shoot_config["time_of_day"]),
                "camera": random.choice(self.cameras),
                "location": random.choice(self.locations)
            })
            
            shoot_id += 1
        
        return shoots
    
    def _calculate_time_of_day(self, dt: datetime) -> str:
        """Determine time of day category from datetime."""
        hour = dt.hour
        if 5 <= hour < 7:
            return "sunrise"
        elif 7 <= hour < 11:
            return "morning"
        elif 11 <= hour < 16:
            return "afternoon"
        elif 16 <= hour < 18:
            return "golden_hour"
        elif 18 <= hour < 20:
            return "blue_hour"
        else:
            return "night"
    
    def _generate_exif_for_genre(self, genre: str, lens_type: str) -> Dict:
        """Generate realistic EXIF data based on shooting genre."""
        exif = {}
        
        if genre in ["portrait", "wedding"]:
            # Portrait: wide aperture, moderate focal length
            exif["focal_length"] = np.random.choice([35, 50, 85], p=[0.2, 0.5, 0.3])
            exif["aperture"] = np.random.choice([1.4, 1.8, 2.0, 2.8], p=[0.2, 0.4, 0.2, 0.2])
            exif["iso"] = np.random.choice([100, 200, 400, 800, 1600], p=[0.2, 0.3, 0.2, 0.2, 0.1])
            exif["shutter_speed"] = np.random.choice([1/125, 1/160, 1/200, 1/250, 1/320], p=[0.2, 0.2, 0.3, 0.2, 0.1])
            
        elif genre == "landscape":
            # Landscape: narrow aperture, wide focal length
            exif["focal_length"] = np.random.choice([14, 16, 24, 35], p=[0.15, 0.25, 0.4, 0.2])
            exif["aperture"] = np.random.choice([8.0, 11.0, 13.0, 16.0], p=[0.3, 0.4, 0.2, 0.1])
            exif["iso"] = np.random.choice([100, 200, 400], p=[0.6, 0.3, 0.1])
            exif["shutter_speed"] = np.random.choice([1/60, 1/125, 1/250, 1.0, 5.0], p=[0.3, 0.3, 0.2, 0.1, 0.1])
            
        elif genre == "event":
            # Event: variable settings, often higher ISO
            exif["focal_length"] = np.random.choice([24, 35, 50, 70, 85], p=[0.2, 0.2, 0.2, 0.2, 0.2])
            exif["aperture"] = np.random.choice([2.8, 4.0, 5.6], p=[0.5, 0.3, 0.2])
            exif["iso"] = np.random.choice([800, 1600, 3200, 6400], p=[0.2, 0.3, 0.3, 0.2])
            exif["shutter_speed"] = np.random.choice([1/60, 1/125, 1/160, 1/200], p=[0.3, 0.3, 0.2, 0.2])
            
        elif genre == "street":
            # Street: moderate settings, documentary style
            exif["focal_length"] = np.random.choice([24, 35, 50], p=[0.3, 0.4, 0.3])
            exif["aperture"] = np.random.choice([4.0, 5.6, 8.0], p=[0.3, 0.4, 0.3])
            exif["iso"] = np.random.choice([200, 400, 800, 1600], p=[0.2, 0.3, 0.3, 0.2])
            exif["shutter_speed"] = np.random.choice([1/125, 1/250, 1/320, 1/500], p=[0.2, 0.3, 0.3, 0.2])
            
        else:  # nature, wildlife
            exif["focal_length"] = np.random.choice([100, 200, 300, 400], p=[0.2, 0.3, 0.3, 0.2])
            exif["aperture"] = np.random.choice([4.0, 5.6, 8.0], p=[0.3, 0.5, 0.2])
            exif["iso"] = np.random.choice([400, 800, 1600, 3200], p=[0.2, 0.3, 0.3, 0.2])
            exif["shutter_speed"] = np.random.choice([1/500, 1/1000, 1/2000, 1/4000], p=[0.2, 0.3, 0.3, 0.2])
        
        return exif
    
    def _get_user_type_for_photo(self, photo_index: int, total_photos: int) -> str:
        """Assign user organization type to photo."""
        # 40% disorganized, 40% partial, 20% power user
        ratio = photo_index / total_photos
        if ratio < 0.4:
            return "disorganized"
        elif ratio < 0.8:
            return "partial"
        else:
            return "power_user"
    
    def _generate_organizational_data(self, user_type: str, shoot: Dict, 
                                      photo_num_in_shoot: int, capture_date: datetime) -> Dict:
        """Generate organizational metadata based on user type."""
        org_data = {}
        
        # Generate folder path
        if user_type == "disorganized":
            org_data["folder_path"] = f"/{capture_date.year}/{capture_date.strftime('%Y-%m-%d')}/"
            org_data["collections"] = []
            org_data["keywords"] = []
            if random.random() < 0.3:  # 30% have 1 keyword
                org_data["keywords"] = [random.choice(list(self.keywords_by_genre[shoot["genres"][0]]))]
            
        elif user_type == "partial":
            # Sometimes organized folders
            if random.random() < 0.6:
                org_data["folder_path"] = f"/{capture_date.year}/{capture_date.strftime('%Y-%m')}-{shoot['shoot_type'].title()}/"
            else:
                org_data["folder_path"] = f"/{capture_date.year}/"
            
            # Few collections
            if random.random() < 0.3:
                org_data["collections"] = [shoot["shoot_type"].title()]
            else:
                org_data["collections"] = []
            
            # Some keywords (2-5)
            num_keywords = random.randint(2, 5)
            org_data["keywords"] = random.sample(
                self.keywords_by_genre[shoot["genres"][0]], 
                min(num_keywords, len(self.keywords_by_genre[shoot["genres"][0]]))
            )
            
        else:  # power_user
            # Well-structured folders
            shoot_name = f"{capture_date.strftime('%Y-%m-%d')}-{shoot['shoot_type'].title()}"
            if shoot["location"] and shoot["location"] != "None":
                shoot_name += f"-{shoot['location'].split(',')[0]}"
            org_data["folder_path"] = f"/{capture_date.year}/{capture_date.strftime('%Y-%m')}/{shoot_name}/"
            
            # Multiple collections
            collections = [shoot["shoot_type"].title(), f"Best of {capture_date.year}"]
            if shoot["location"]:
                collections.append(f"Location: {shoot['location']}")
            org_data["collections"] = collections
            
            # Extensive keywords (8-15)
            base_keywords = self.keywords_by_genre[shoot["genres"][0]].copy()
            additional = ["sharp", "keeper", "edited", "export"]
            all_keywords = base_keywords + additional
            num_keywords = random.randint(8, min(15, len(all_keywords)))
            org_data["keywords"] = random.sample(all_keywords, num_keywords)
        
        # Color labels
        if user_type == "power_user":
            org_data["color_label"] = np.random.choice(
                [None, "red", "yellow", "green"], 
                p=[0.5, 0.2, 0.2, 0.1]
            )
        else:
            org_data["color_label"] = np.random.choice(
                [None, "red", "yellow"], 
                p=[0.8, 0.1, 0.1]
            )
        
        # Star ratings
        if user_type == "disorganized":
            org_data["star_rating"] = np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])
        elif user_type == "partial":
            org_data["star_rating"] = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.2, 0.2, 0.15, 0.05])
        else:
            org_data["star_rating"] = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1])
        
        # Flag status
        if user_type == "power_user":
            org_data["flag_status"] = np.random.choice(
                ["unflagged", "picked", "rejected"], 
                p=[0.3, 0.5, 0.2]
            )
        else:
            org_data["flag_status"] = np.random.choice(
                ["unflagged", "picked", "rejected"], 
                p=[0.7, 0.2, 0.1]
            )
        
        # Edit data
        if user_type == "disorganized":
            org_data["has_edits"] = random.random() < 0.2
        elif user_type == "partial":
            org_data["has_edits"] = random.random() < 0.5
        else:
            org_data["has_edits"] = random.random() < 0.8
        
        if org_data["has_edits"]:
            org_data["edit_count"] = np.random.poisson(5) + 1  # Poisson distribution, min 1
            days_to_edit = np.random.exponential(7) + 1  # Edit within ~7 days typically
            org_data["last_modified_date"] = capture_date + timedelta(days=days_to_edit)
        else:
            org_data["edit_count"] = 0
            org_data["last_modified_date"] = capture_date
        
        return org_data
    
    def generate_catalog(self) -> pd.DataFrame:
        """Generate complete synthetic Lightroom catalog."""
        print("Generating photo shoots...")
        shoots = self._generate_photo_shoots()
        
        photos = []
        photo_id = 1
        
        print(f"Generating {self.num_photos} photos across {len(shoots)} shoots...")
        
        for shoot in shoots:
            shoot_photo_count = min(shoot["photo_count"], self.num_photos - len(photos))
            if shoot_photo_count <= 0:
                break
            
            camera = shoot["camera"]
            file_ext = self.file_extensions[camera]
            base_date = shoot["date"]
            
            # Select lens for shoot
            lens_category = random.choice(shoot["preferred_lenses"])
            lens = random.choice(self.lenses[lens_category])
            
            for i in range(shoot_photo_count):
                # Time variation within shoot
                photo_time = base_date + timedelta(
                    hours=random.randint(0, shoot["duration_hours"]),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                
                # Determine user type for this photo
                user_type = self._get_user_type_for_photo(len(photos), self.num_photos)
                
                # Generate EXIF
                genre = shoot["genres"][0]
                exif = self._generate_exif_for_genre(genre, lens_category)
                
                # File info
                is_raw = random.random() < 0.85  # 85% RAW files
                if is_raw:
                    filename = f"IMG_{photo_id:04d}.{file_ext}"
                    file_size = np.random.uniform(15, 45)
                else:
                    filename = f"IMG_{photo_id:04d}.JPG"
                    file_size = np.random.uniform(2, 10)
                
                # Organizational data
                org_data = self._generate_organizational_data(
                    user_type, shoot, i, photo_time
                )
                
                photo = {
                    "photo_id": f"photo_{photo_id:06d}",
                    "filename": filename,
                    "capture_date": photo_time,
                    "file_type": file_ext if is_raw else "JPG",
                    "file_size_mb": round(file_size, 2),
                    "camera_model": camera,
                    "lens_model": lens,
                    "focal_length": exif["focal_length"],
                    "aperture": exif["aperture"],
                    "shutter_speed": exif["shutter_speed"],
                    "iso": int(exif["iso"]),
                    "location": shoot["location"],
                    "time_of_day": self._calculate_time_of_day(photo_time),
                    "folder_path": org_data["folder_path"],
                    "collections": json.dumps(org_data["collections"]),
                    "keywords": json.dumps(org_data["keywords"]),
                    "color_label": org_data["color_label"],
                    "star_rating": org_data["star_rating"],
                    "flag_status": org_data["flag_status"],
                    "has_edits": org_data["has_edits"],
                    "edit_count": org_data["edit_count"],
                    "last_modified_date": org_data["last_modified_date"],
                    "shoot_id": shoot["shoot_id"],
                    "genre": genre
                }
                
                photos.append(photo)
                photo_id += 1
                
                if len(photos) >= self.num_photos:
                    break
            
            if len(photos) >= self.num_photos:
                break
        
        df = pd.DataFrame(photos)
        print(f"\n✓ Generated {len(df)} photos")
        print(f"✓ Date range: {df['capture_date'].min()} to {df['capture_date'].max()}")
        print(f"✓ {len(shoots)} photo shoots")
        
        return df


def main():
    """Generate and save synthetic catalog data."""
    print("=" * 60)
    print("Lightroom Catalog Synthetic Data Generator")
    print("=" * 60)
    
    # Generate catalog
    generator = LightroomCatalogGenerator(num_photos=5000)
    catalog_df = generator.generate_catalog()
    
    # Save to CSV
    output_file = "lightroom_catalog_synthetic.csv"
    catalog_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved catalog to: {output_file}")
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("CATALOG SUMMARY")
    print("=" * 60)
    print(f"Total Photos: {len(catalog_df):,}")
    print(f"Total Size: {catalog_df['file_size_mb'].sum():.2f} GB")
    print(f"\nFile Types:")
    print(catalog_df['file_type'].value_counts())
    print(f"\nCamera Bodies:")
    print(catalog_df['camera_model'].value_counts())
    print(f"\nGenres:")
    print(catalog_df['genre'].value_counts())
    print(f"\nStar Ratings:")
    print(catalog_df['star_rating'].value_counts().sort_index())
    
    # Organizational health snapshot
    total_photos = len(catalog_df)
    photos_with_keywords = catalog_df['keywords'].apply(lambda x: len(json.loads(x)) > 0).sum()
    photos_in_collections = catalog_df['collections'].apply(lambda x: len(json.loads(x)) > 0).sum()
    photos_rated = (catalog_df['star_rating'] > 0).sum()
    photos_edited = catalog_df['has_edits'].sum()
    
    print(f"\nORGANIZATIONAL HEALTH:")
    print(f"  Photos with keywords: {photos_with_keywords} ({photos_with_keywords/total_photos*100:.1f}%)")
    print(f"  Photos in collections: {photos_in_collections} ({photos_in_collections/total_photos*100:.1f}%)")
    print(f"  Photos rated: {photos_rated} ({photos_rated/total_photos*100:.1f}%)")
    print(f"  Photos edited: {photos_edited} ({photos_edited/total_photos*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✓ Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()