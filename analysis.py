"""
Pattern Analysis Engine for Lightroom Catalogs

Detects organizational patterns, workflow inefficiencies, and shooting styles
to power Smart Collection recommendations and health scoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
from collections import Counter
from sklearn.cluster import KMeans


class CatalogAnalyzer:
    """Analyzes Lightroom catalog patterns and computes health metrics."""
    
    def __init__(self, catalog_df: pd.DataFrame):
        self.df = catalog_df.copy()
        self._prepare_data()
        self.analysis_results = {}
        
    def _prepare_data(self):
        """Prepare data for analysis."""
        # Parse JSON fields
        self.df['keywords_list'] = self.df['keywords'].apply(json.loads)
        self.df['collections_list'] = self.df['collections'].apply(json.loads)
        
        # Calculate derived metrics
        self.df['keyword_count'] = self.df['keywords_list'].apply(len)
        self.df['collection_count'] = self.df['collections_list'].apply(len)
        self.df['is_rated'] = self.df['star_rating'] > 0
        self.df['is_in_collection'] = self.df['collection_count'] > 0
        self.df['has_keywords'] = self.df['keyword_count'] > 0
        
        # Time-based metrics
        self.df['capture_date'] = pd.to_datetime(self.df['capture_date'])
        self.df['last_modified_date'] = pd.to_datetime(self.df['last_modified_date'])
        self.df['days_to_edit'] = (self.df['last_modified_date'] - self.df['capture_date']).dt.days
        
        # Age of photos
        current_date = datetime.now()
        self.df['photo_age_days'] = (current_date - self.df['capture_date']).dt.days
        
        # Is photo abandoned? (old, never touched)
        self.df['is_abandoned'] = (
            (self.df['photo_age_days'] > 180) & 
            (~self.df['has_edits']) & 
            (~self.df['is_rated']) & 
            (~self.df['has_keywords'])
        )
        
        # Is photo orphan? (no organization at all)
        self.df['is_orphan'] = (
            (~self.df['has_keywords']) & 
            (~self.df['is_in_collection']) & 
            (~self.df['is_rated'])
        )
        
    def analyze_keywords(self) -> Dict[str, Any]:
        """Analyze keyword usage patterns."""
        results = {
            'total_photos': len(self.df),
            'photos_with_keywords': self.df['has_keywords'].sum(),
            'photos_without_keywords': (~self.df['has_keywords']).sum(),
            'pct_with_keywords': (self.df['has_keywords'].sum() / len(self.df)) * 100,
            'pct_without_keywords': ((~self.df['has_keywords']).sum() / len(self.df)) * 100,
        }
        
        # Keyword count distribution
        keyword_dist = self.df['keyword_count'].value_counts().sort_index().to_dict()
        results['keyword_distribution'] = keyword_dist
        
        # Categorize by keyword coverage
        results['no_keywords'] = (self.df['keyword_count'] == 0).sum()
        results['minimal_keywords'] = ((self.df['keyword_count'] >= 1) & (self.df['keyword_count'] <= 3)).sum()
        results['moderate_keywords'] = ((self.df['keyword_count'] >= 4) & (self.df['keyword_count'] <= 7)).sum()
        results['extensive_keywords'] = (self.df['keyword_count'] >= 8).sum()
        
        # Most common keywords
        all_keywords = []
        for kw_list in self.df['keywords_list']:
            all_keywords.extend(kw_list)
        
        keyword_counter = Counter(all_keywords)
        results['top_keywords'] = dict(keyword_counter.most_common(20))
        results['total_unique_keywords'] = len(keyword_counter)
        
        # Keyword consistency over time (by quarter)
        self.df['quarter'] = self.df['capture_date'].dt.to_period('Q')
        keyword_by_quarter = self.df.groupby('quarter')['has_keywords'].mean() * 100
        results['keyword_consistency_by_quarter'] = keyword_by_quarter.to_dict()
        
        # Orphan photos (no keywords, no collections, no ratings)
        results['orphan_photos'] = self.df['is_orphan'].sum()
        results['pct_orphan'] = (self.df['is_orphan'].sum() / len(self.df)) * 100
        
        return results
    
    def analyze_folders(self) -> Dict[str, Any]:
        """Analyze folder organization patterns."""
        results = {}
        
        # Folder count and distribution
        folder_counts = self.df['folder_path'].value_counts()
        results['total_folders'] = len(folder_counts)
        results['photos_per_folder'] = folder_counts.to_dict()
        results['avg_photos_per_folder'] = folder_counts.mean()
        results['median_photos_per_folder'] = folder_counts.median()
        
        # Identify overstuffed folders (>500 photos)
        overstuffed = folder_counts[folder_counts > 500]
        results['overstuffed_folders'] = len(overstuffed)
        results['overstuffed_folder_list'] = overstuffed.to_dict()
        
        # Folder depth analysis
        self.df['folder_depth'] = self.df['folder_path'].apply(lambda x: len([p for p in x.split('/') if p]))
        results['avg_folder_depth'] = self.df['folder_depth'].mean()
        results['max_folder_depth'] = self.df['folder_depth'].max()
        results['min_folder_depth'] = self.df['folder_depth'].min()
        
        # Folder depth distribution
        depth_dist = self.df['folder_depth'].value_counts().sort_index().to_dict()
        results['folder_depth_distribution'] = depth_dist
        
        # Naming convention detection
        date_based = self.df['folder_path'].str.contains(r'\d{4}').sum()
        results['date_based_folders'] = date_based
        results['pct_date_based'] = (date_based / len(self.df)) * 100
        
        # Flat vs hierarchical
        flat_structure = (self.df['folder_depth'] <= 2).sum()
        results['flat_structure_photos'] = flat_structure
        results['hierarchical_structure_photos'] = len(self.df) - flat_structure
        results['pct_flat'] = (flat_structure / len(self.df)) * 100
        
        return results
    
    def analyze_collections(self) -> Dict[str, Any]:
        """Analyze collection usage patterns."""
        results = {}
        
        # Collection usage
        results['photos_in_collections'] = self.df['is_in_collection'].sum()
        results['photos_not_in_collections'] = (~self.df['is_in_collection']).sum()
        results['pct_in_collections'] = (self.df['is_in_collection'].sum() / len(self.df)) * 100
        
        # Total unique collections
        all_collections = []
        for coll_list in self.df['collections_list']:
            all_collections.extend(coll_list)
        
        collection_counter = Counter(all_collections)
        results['total_collections'] = len(collection_counter)
        results['collection_usage'] = dict(collection_counter.most_common(20))
        
        # Average photos per collection
        if len(collection_counter) > 0:
            results['avg_photos_per_collection'] = len(all_collections) / len(collection_counter)
        else:
            results['avg_photos_per_collection'] = 0
        
        # Collection overlap (photos in multiple collections)
        multi_collection = (self.df['collection_count'] > 1).sum()
        results['photos_in_multiple_collections'] = multi_collection
        results['pct_multi_collection'] = (multi_collection / len(self.df)) * 100
        
        # Collection distribution
        coll_dist = self.df['collection_count'].value_counts().sort_index().to_dict()
        results['collection_count_distribution'] = coll_dist
        
        return results
    
    def analyze_ratings_flags(self) -> Dict[str, Any]:
        """Analyze rating and flag patterns."""
        results = {}
        
        # Star rating distribution
        rating_dist = self.df['star_rating'].value_counts().sort_index().to_dict()
        results['rating_distribution'] = rating_dist
        
        # Rating stats
        results['photos_rated'] = self.df['is_rated'].sum()
        results['photos_unrated'] = (~self.df['is_rated']).sum()
        results['pct_rated'] = (self.df['is_rated'].sum() / len(self.df)) * 100
        results['avg_rating'] = self.df[self.df['is_rated']]['star_rating'].mean()
        
        # Flag status distribution
        flag_dist = self.df['flag_status'].value_counts().to_dict()
        results['flag_distribution'] = flag_dist
        results['pct_picked'] = (self.df['flag_status'] == 'picked').sum() / len(self.df) * 100
        results['pct_rejected'] = (self.df['flag_status'] == 'rejected').sum() / len(self.df) * 100
        results['pct_unflagged'] = (self.df['flag_status'] == 'unflagged').sum() / len(self.df) * 100
        
        # Culling workflow detection
        reject_rate = (self.df['flag_status'] == 'rejected').sum() / len(self.df)
        results['reject_rate'] = reject_rate * 100
        results['has_culling_workflow'] = reject_rate > 0.15  # >15% rejection suggests active culling
        
        # Color label usage
        color_dist = self.df['color_label'].value_counts().to_dict()
        results['color_label_distribution'] = color_dist
        results['uses_color_labels'] = self.df['color_label'].notna().sum() > 0
        results['pct_color_labeled'] = (self.df['color_label'].notna().sum() / len(self.df)) * 100
        
        return results
    
    def analyze_shooting_style(self) -> Dict[str, Any]:
        """Detect shooting style and genre preferences using ML clustering."""
        results = {}
        
        # Genre distribution
        genre_dist = self.df['genre'].value_counts()
        results['genre_distribution'] = genre_dist.to_dict()
        results['primary_genre'] = genre_dist.index[0]
        results['genre_percentages'] = (genre_dist / len(self.df) * 100).to_dict()
        
        # ML-POWERED: K-Means clustering on EXIF data to detect shooting styles
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features for clustering
        exif_features = self.df[['focal_length', 'aperture', 'iso']].copy()
        exif_features = exif_features.dropna()
        
        if len(exif_features) > 100:  # Only cluster if enough data
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(exif_features)
            
            # K-means clustering to find shooting style groups
            n_clusters = min(5, len(exif_features) // 100)  # 5 clusters or less
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Analyze cluster characteristics
            exif_features['cluster'] = clusters
            cluster_analysis = {}
            
            for cluster_id in range(n_clusters):
                cluster_data = exif_features[exif_features['cluster'] == cluster_id]
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'avg_focal_length': float(cluster_data['focal_length'].mean()),
                    'avg_aperture': float(cluster_data['aperture'].mean()),
                    'avg_iso': float(cluster_data['iso'].mean()),
                    'style_description': self._describe_cluster_style(
                        cluster_data['focal_length'].mean(),
                        cluster_data['aperture'].mean(),
                        cluster_data['iso'].mean()
                    )
                }
            
            results['ml_detected_styles'] = cluster_analysis
            results['dominant_style_cluster'] = int(exif_features['cluster'].mode()[0])
        
        # Focal length preferences
        focal_dist = self.df['focal_length'].value_counts().sort_index()
        results['focal_length_distribution'] = focal_dist.to_dict()
        results['most_used_focal_length'] = self.df['focal_length'].mode()[0]
        results['avg_focal_length'] = self.df['focal_length'].mean()
        
        # Aperture preferences
        aperture_dist = self.df['aperture'].value_counts().sort_index()
        results['aperture_distribution'] = aperture_dist.to_dict()
        results['most_used_aperture'] = self.df['aperture'].mode()[0]
        results['avg_aperture'] = self.df['aperture'].mean()
        
        # Detect portrait shooter (focal length > 50mm, aperture < 4)
        portrait_shots = ((self.df['focal_length'] >= 50) & (self.df['aperture'] <= 4.0)).sum()
        results['portrait_style_shots'] = portrait_shots
        results['pct_portrait_style'] = (portrait_shots / len(self.df)) * 100
        
        # Detect landscape shooter (focal length < 35mm, aperture > 5.6)
        landscape_shots = ((self.df['focal_length'] <= 35) & (self.df['aperture'] >= 5.6)).sum()
        results['landscape_style_shots'] = landscape_shots
        results['pct_landscape_style'] = (landscape_shots / len(self.df)) * 100
        
        # ISO preferences
        iso_dist = self.df['iso'].value_counts().sort_index()
        results['iso_distribution'] = iso_dist.to_dict()
        results['avg_iso'] = self.df['iso'].mean()
        results['high_iso_shots'] = (self.df['iso'] > 3200).sum()
        results['pct_high_iso'] = (self.df['iso'] > 3200).sum() / len(self.df) * 100
        
        # Time of day preferences
        time_dist = self.df['time_of_day'].value_counts()
        results['time_of_day_distribution'] = time_dist.to_dict()
        results['preferred_time'] = time_dist.index[0]
        
        # Golden hour shooter detection
        golden_shots = (self.df['time_of_day'] == 'golden_hour').sum()
        results['golden_hour_shots'] = golden_shots
        results['pct_golden_hour'] = (golden_shots / len(self.df)) * 100
        results['is_golden_hour_shooter'] = (golden_shots / len(self.df)) > 0.25
        
        # Equipment usage
        camera_dist = self.df['camera_model'].value_counts()
        results['camera_distribution'] = camera_dist.to_dict()
        results['primary_camera'] = camera_dist.index[0]
        results['camera_count'] = len(camera_dist)
        
        lens_dist = self.df['lens_model'].value_counts()
        results['lens_distribution'] = lens_dist.to_dict()
        results['primary_lens'] = lens_dist.index[0]
        results['lens_count'] = len(lens_dist)
        
        # Camera + lens combinations
        cam_lens = self.df.groupby(['camera_model', 'lens_model']).size().sort_values(ascending=False)
        results['top_camera_lens_combos'] = cam_lens.head(10).to_dict()
        
        return results
    
    def _describe_cluster_style(self, focal_length: float, aperture: float, iso: float) -> str:
        """Describe shooting style based on cluster characteristics."""
        if focal_length > 70 and aperture < 4.0:
            return "Portrait/Telephoto - shallow DOF"
        elif focal_length < 35 and aperture > 5.6:
            return "Landscape/Wide - deep DOF"
        elif iso > 1600:
            return "Event/Low-light - high ISO"
        elif focal_length >= 35 and focal_length <= 70:
            return "Versatile/Documentary"
        else:
            return "Mixed shooting style"
    
    def analyze_workflow_efficiency(self) -> Dict[str, Any]:
        """Analyze workflow patterns and efficiency."""
        results = {}
        
        # Edit completion rate
        results['photos_edited'] = self.df['has_edits'].sum()
        results['photos_unedited'] = (~self.df['has_edits']).sum()
        results['edit_rate'] = (self.df['has_edits'].sum() / len(self.df)) * 100
        
        # Edit count stats
        edited_photos = self.df[self.df['has_edits']]
        if len(edited_photos) > 0:
            results['avg_edits_per_photo'] = edited_photos['edit_count'].mean()
            results['max_edits'] = edited_photos['edit_count'].max()
            results['edit_count_distribution'] = edited_photos['edit_count'].value_counts().sort_index().head(20).to_dict()
        else:
            results['avg_edits_per_photo'] = 0
            results['max_edits'] = 0
            results['edit_count_distribution'] = {}
        
        # Time to edit analysis
        edited_with_time = self.df[self.df['has_edits'] & (self.df['days_to_edit'] >= 0)]
        if len(edited_with_time) > 0:
            results['avg_days_to_edit'] = edited_with_time['days_to_edit'].mean()
            results['median_days_to_edit'] = edited_with_time['days_to_edit'].median()
            
            # Categorize editing speed
            quick_edit = (edited_with_time['days_to_edit'] <= 7).sum()
            moderate_edit = ((edited_with_time['days_to_edit'] > 7) & (edited_with_time['days_to_edit'] <= 30)).sum()
            slow_edit = (edited_with_time['days_to_edit'] > 30).sum()
            
            results['quick_edits'] = quick_edit
            results['moderate_edits'] = moderate_edit
            results['slow_edits'] = slow_edit
        else:
            results['avg_days_to_edit'] = 0
            results['median_days_to_edit'] = 0
        
        # Abandoned photos (captured >6 months ago, never touched)
        results['abandoned_photos'] = self.df['is_abandoned'].sum()
        results['pct_abandoned'] = (self.df['is_abandoned'].sum() / len(self.df)) * 100
        
        # Recent activity (last 30 days)
        recent_imports = (self.df['photo_age_days'] <= 30).sum()
        results['recent_imports'] = recent_imports
        results['pct_recent'] = (recent_imports / len(self.df)) * 100
        
        # Batch editing detection (multiple photos edited same day)
        if len(edited_photos) > 0:
            edit_dates = edited_photos['last_modified_date'].dt.date.value_counts()
            batch_edits = (edit_dates > 10).sum()  # Days with 10+ edits
            results['batch_edit_days'] = batch_edits
            results['avg_photos_per_edit_session'] = edit_dates.mean()
        else:
            results['batch_edit_days'] = 0
            results['avg_photos_per_edit_session'] = 0
        
        # Shooting frequency
        self.df['capture_month'] = self.df['capture_date'].dt.to_period('M')
        monthly_captures = self.df.groupby('capture_month').size()
        results['avg_photos_per_month'] = monthly_captures.mean()
        results['most_active_month'] = monthly_captures.idxmax().strftime('%Y-%m')
        results['least_active_month'] = monthly_captures.idxmin().strftime('%Y-%m')
        
        # Identify needs attention (unrated AND unedited AND old)
        needs_attention = (
            (~self.df['is_rated']) & 
            (~self.df['has_edits']) & 
            (self.df['photo_age_days'] > 30)
        ).sum()
        results['needs_attention'] = needs_attention
        results['pct_needs_attention'] = (needs_attention / len(self.df)) * 100
        
        return results
    
    def calculate_health_score(self) -> Dict[str, Any]:
        """Calculate overall organizational health score (0-100)."""
        scores = {}
        
        # Keyword coverage (30 points)
        keyword_score = (self.df['has_keywords'].sum() / len(self.df)) * 30
        scores['keyword_score'] = round(keyword_score, 2)
        
        # Collection usage (20 points)
        collection_score = (self.df['is_in_collection'].sum() / len(self.df)) * 20
        scores['collection_score'] = round(collection_score, 2)
        
        # Rating consistency (20 points)
        rating_score = (self.df['is_rated'].sum() / len(self.df)) * 20
        scores['rating_score'] = round(rating_score, 2)
        
        # Folder structure (15 points)
        # Penalize overstuffed folders and too-flat structure
        folder_counts = self.df['folder_path'].value_counts()
        overstuffed_penalty = min((folder_counts > 500).sum() * 2, 10)
        flat_penalty = min(((self.df['folder_depth'] <= 1).sum() / len(self.df)) * 10, 5)
        folder_score = max(15 - overstuffed_penalty - flat_penalty, 0)
        scores['folder_score'] = round(folder_score, 2)
        
        # Edit completion rate (15 points)
        edit_score = (self.df['has_edits'].sum() / len(self.df)) * 15
        scores['edit_score'] = round(edit_score, 2)
        
        # Total health score
        total_score = sum(scores.values())
        scores['total_health_score'] = round(total_score, 2)
        
        # Health category
        if total_score >= 76:
            scores['health_category'] = 'Excellent'
            scores['health_color'] = 'green'
        elif total_score >= 51:
            scores['health_category'] = 'Good'
            scores['health_color'] = 'yellow'
        else:
            scores['health_category'] = 'Needs Improvement'
            scores['health_color'] = 'red'
        
        return scores
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete catalog analysis."""
        print("Running catalog analysis...")
        
        self.analysis_results = {
            'catalog_overview': {
                'total_photos': len(self.df),
                'date_range': {
                    'oldest': self.df['capture_date'].min().strftime('%Y-%m-%d'),
                    'newest': self.df['capture_date'].max().strftime('%Y-%m-%d'),
                },
                'total_size_gb': round(self.df['file_size_mb'].sum() / 1024, 2),
                'file_type_distribution': self.df['file_type'].value_counts().to_dict(),
            },
            'keyword_analysis': self.analyze_keywords(),
            'folder_analysis': self.analyze_folders(),
            'collection_analysis': self.analyze_collections(),
            'rating_flag_analysis': self.analyze_ratings_flags(),
            'shooting_style': self.analyze_shooting_style(),
            'workflow_efficiency': self.analyze_workflow_efficiency(),
            'health_score': self.calculate_health_score(),
        }
        
        print("✓ Analysis complete!")
        return self.analysis_results
    
    def get_catalog_dataframe(self) -> pd.DataFrame:
        """Return processed catalog dataframe."""
        return self.df
    
    def export_analysis(self, filepath: str = 'analysis_results.json'):
        """Export analysis results to JSON."""
        # Convert any remaining Period objects and numpy types
        def convert_types(obj):
            if isinstance(obj, pd.Period):
                return str(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                # Convert Period keys to strings, recurse on values
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        cleaned_results = convert_types(self.analysis_results)
        
        with open(filepath, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
        
        print(f"✓ Analysis exported to: {filepath}")


def main():
    """Load catalog and run analysis."""
    print("=" * 60)
    print("Lightroom Catalog Pattern Analysis")
    print("=" * 60)
    
    # Load catalog
    print("\nLoading catalog...")
    try:
        catalog_df = pd.read_csv('lightroom_catalog_synthetic.csv')
        print(f"✓ Loaded {len(catalog_df):,} photos")
    except FileNotFoundError:
        print("Error: lightroom_catalog_synthetic.csv not found!")
        print("Please run generate_data.py first.")
        return
    
    # Run analysis
    analyzer = CatalogAnalyzer(catalog_df)
    results = analyzer.run_full_analysis()
    
    # Display key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    health = results['health_score']
    print(f"\n📊 ORGANIZATIONAL HEALTH SCORE: {health['total_health_score']:.1f}/100")
    print(f"   Status: {health['health_category']}")
    print(f"   - Keywords: {health['keyword_score']:.1f}/30")
    print(f"   - Collections: {health['collection_score']:.1f}/20")
    print(f"   - Ratings: {health['rating_score']:.1f}/20")
    print(f"   - Folders: {health['folder_score']:.1f}/15")
    print(f"   - Edits: {health['edit_score']:.1f}/15")
    
    keywords = results['keyword_analysis']
    print(f"\n🏷️  KEYWORDS:")
    print(f"   - {keywords['pct_without_keywords']:.1f}% of photos lack keywords")
    print(f"   - {keywords['orphan_photos']:,} orphan photos (no keywords/collections/ratings)")
    
    workflow = results['workflow_efficiency']
    print(f"\n⚡ WORKFLOW:")
    print(f"   - {workflow['edit_rate']:.1f}% of photos edited")
    print(f"   - {workflow['abandoned_photos']:,} abandoned photos")
    print(f"   - Avg {workflow['avg_days_to_edit']:.1f} days from capture to edit")
    
    style = results['shooting_style']
    print(f"\n📸 SHOOTING STYLE:")
    print(f"   - Primary genre: {style['primary_genre']}")
    print(f"   - Favorite lens: {style['primary_lens']}")
    print(f"   - Most used focal length: {style['most_used_focal_length']}mm")
    
    # Export results
    analyzer.export_analysis()
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()