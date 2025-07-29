"""
SLAM+ADAS Data Pipeline
Complete data loading and preprocessing system for KITTI, PKLot, and Waymo datasets
"""
TF_ENABLE_ONEDNN_OPTS=0
import os
import numpy as np
import cv2
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
from enum import Enum
import tensorflow as tf

# Data structures for different sensor types
@dataclass
class CameraData:
    """Camera sensor data container"""
    left_image: np.ndarray
    right_image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    camera_id: str = "cam_0"

@dataclass
class LiDARData:
    """LiDAR sensor data container"""
    points: np.ndarray  # N x 4 (x, y, z, intensity)
    timestamp: float = 0.0 
    frame_id: str = "lidar_0"

@dataclass
class IMUData:
    """IMU sensor data container"""
    acceleration: np.ndarray  # 3D acceleration
    angular_velocity: np.ndarray  # 3D angular velocity
    orientation: Optional[np.ndarray] = None  # Quaternion
    timestamp: float = 0.0

@dataclass
class GPSData:
    """GPS sensor data container"""
    latitude: float
    longitude: float
    altitude: float
    timestamp: float = 0.0

@dataclass
class ObjectLabel:
    """3D object detection label"""
    class_name: str
    bbox_2d: np.ndarray  # [x1, y1, x2, y2]
    bbox_3d: Dict[str, Any]  # 3D bounding box parameters
    confidence: float = 1.0

@dataclass
class SensorFrame:
    """Complete sensor frame containing all modalities"""
    frame_id: int
    timestamp: float
    camera_data: CameraData
    lidar_data: Optional[LiDARData] = None
    imu_data: Optional[IMUData] = None
    gps_data: Optional[GPSData] = None
    labels: List[ObjectLabel] = None
    calibration: Dict[str, np.ndarray] = None

class DatasetType(Enum):
    KITTI_OBJECT = "kitti_object"
    KITTI_ODOMETRY = "kitti_odometry" 
    KITTI_RAW = "kitti_raw"
    PKLOT = "pklot"
    WAYMO = "waymo"

class KITTIObjectLoader:
    """KITTI Object Detection Dataset Loader"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.training_path = self.data_path / "training"
        
        # Verify paths exist
        required_paths = ["image_2", "image_3", "label_2", "calib", "velodyne"]
        for path_name in required_paths:
            path = self.training_path / path_name
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")
        
        # Get file lists
        self.image_files = sorted(list((self.training_path / "image_2").glob("*.png")))
        self.num_samples = len(self.image_files)
        print(f"Found {self.num_samples} KITTI object samples")
    
    def load_calibration(self, idx: int) -> Dict[str, np.ndarray]:
        """Load calibration matrices for a specific frame"""
        calib_file = self.training_path / "calib" / f"{idx:06d}.txt"
        
        calib_data = {}
        with open(calib_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    key, values = line.split(':', 1)
                    calib_data[key] = np.array([float(x) for x in values.split()])
        
        # Reshape matrices
        if 'P2' in calib_data:
            calib_data['P2'] = calib_data['P2'].reshape(3, 4)
        if 'P3' in calib_data:
            calib_data['P3'] = calib_data['P3'].reshape(3, 4)
        if 'R0_rect' in calib_data:
            calib_data['R0_rect'] = calib_data['R0_rect'].reshape(3, 3)
        if 'Tr_velo_to_cam' in calib_data:
            calib_data['Tr_velo_to_cam'] = calib_data['Tr_velo_to_cam'].reshape(3, 4)
            
        return calib_data
    
    def load_lidar_points(self, idx: int) -> np.ndarray:
        """Load LiDAR point cloud"""
        lidar_file = self.training_path / "velodyne" / f"{idx:06d}.bin"
        
        # Read binary point cloud file
        points = np.fromfile(str(lidar_file), dtype=np.float32)
        points = points.reshape(-1, 4)  # x, y, z, intensity
        
        return points
    
    def load_labels(self, idx: int) -> List[ObjectLabel]:
        """Load 3D object labels"""
        label_file = self.training_path / "label_2" / f"{idx:06d}.txt"
        
        if not label_file.exists():
            return []
        
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 15:
                    # Parse KITTI label format
                    class_name = parts[0]
                    bbox_2d = np.array([float(parts[4]), float(parts[5]), 
                                      float(parts[6]), float(parts[7])])
                    
                    bbox_3d = {
                        'dimensions': np.array([float(parts[8]), float(parts[9]), float(parts[10])]),  # h,w,l
                        'location': np.array([float(parts[11]), float(parts[12]), float(parts[13])]),   # x,y,z
                        'rotation_y': float(parts[14])
                    }
                    
                    label = ObjectLabel(
                        class_name=class_name,
                        bbox_2d=bbox_2d,
                        bbox_3d=bbox_3d
                    )
                    labels.append(label)
        
        return labels
    
    def get_frame(self, idx: int) -> SensorFrame:
        """Load complete sensor frame"""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range (max: {self.num_samples-1})")
        
        # Load images
        left_img_path = self.training_path / "image_2" / f"{idx:06d}.png"
        right_img_path = self.training_path / "image_3" / f"{idx:06d}.png"
        
        left_image = cv2.imread(str(left_img_path))
        right_image = cv2.imread(str(right_img_path)) if right_img_path.exists() else None
        
        camera_data = CameraData(
            left_image=left_image,
            right_image=right_image,
            timestamp=float(idx)  # Use index as timestamp for KITTI
        )
        
        # Load LiDAR
        lidar_points = self.load_lidar_points(idx)
        lidar_data = LiDARData(
            points=lidar_points,
            timestamp=float(idx)
        )
        
        # Load calibration and labels
        calibration = self.load_calibration(idx)
        labels = self.load_labels(idx)
        
        return SensorFrame(
            frame_id=idx,
            timestamp=float(idx),
            camera_data=camera_data,
            lidar_data=lidar_data,
            calibration=calibration,
            labels=labels
        )

class PKLotLoader:
    """PKLot Parking Dataset Loader"""
    
    def __init__(self, data_path: str, split: str = "train"):
        self.data_path = Path(data_path)
        self.split = split
        self.split_path = self.data_path / split
        
        if not self.split_path.exists():
            raise FileNotFoundError(f"Split path not found: {self.split_path}")
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(list(self.split_path.glob(ext)))
        
        self.image_files = sorted(self.image_files)
        self.num_samples = len(self.image_files)
        print(f"Found {self.num_samples} PKLot {split} samples")
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Parse PKLot filename to extract metadata"""
        # PKLot filename format contains parking lot info
        parts = filename.split('_')
        metadata = {
            'lot_id': parts[0] if len(parts) > 0 else 'unknown',
            'weather': 'unknown',
            'occupancy': 'unknown'
        }
        
        # Try to extract weather and occupancy from filename
        filename_lower = filename.lower()
        if 'sunny' in filename_lower:
            metadata['weather'] = 'sunny'
        elif 'overcast' in filename_lower:
            metadata['weather'] = 'overcast'
        elif 'rainy' in filename_lower:
            metadata['weather'] = 'rainy'
            
        return metadata
    
    def get_frame(self, idx: int) -> SensorFrame:
        """Load parking lot image frame"""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range (max: {self.num_samples-1})")
        
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        
        # Parse metadata from filename
        metadata = self.parse_filename(img_path.name)
        
        camera_data = CameraData(
            left_image=image,
            timestamp=float(idx)
        )
        
        return SensorFrame(
            frame_id=idx,
            timestamp=float(idx),
            camera_data=camera_data,
            labels=[]  # PKLot labels would need separate processing
        )

class WaymoLoader:
    """Waymo Open Dataset Loader"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
        # Get TensorFlow record files
        self.tfrecord_files = list(self.data_path.glob("*.tfrecord"))
        if not self.tfrecord_files:
            print("Warning: No .tfrecord files found. Waymo data may need different extraction.")
            self.tfrecord_files = list(self.data_path.glob("*"))  # Get all files
        
        self.num_files = len(self.tfrecord_files)
        self.num_samples = self.num_files  # Add this line to fix the error
        print(f"Found {self.num_files} Waymo record files")
    
    def get_frame(self, idx: int) -> SensorFrame:
        """Load Waymo frame (simplified - needs Waymo SDK for full implementation)"""
        if idx >= self.num_files:
            raise IndexError(f"Index {idx} out of range (max: {self.num_files-1})")
        
        # Placeholder implementation - would need Waymo Open Dataset SDK
        # For now, return empty frame structure
        camera_data = CameraData(
            left_image=np.zeros((1280, 1920, 3), dtype=np.uint8),
            timestamp=float(idx)
        )
        
        return SensorFrame(
            frame_id=idx,
            timestamp=float(idx),
            camera_data=camera_data
        )

class UnifiedDataPipeline:
    """Unified data pipeline for all datasets"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.loaders = {}
        
        # Initialize available loaders
        self._initialize_loaders()
    
    def _initialize_loaders(self):
        """Initialize all available dataset loaders"""
        
        # KITTI Object Detection
        kitti_obj_path = self.base_path / "kitti" / "object_detection"
        if kitti_obj_path.exists():
            try:
                self.loaders[DatasetType.KITTI_OBJECT] = KITTIObjectLoader(str(kitti_obj_path))
                print("✅ KITTI Object Detection loader initialized")
            except Exception as e:
                print(f"❌ Failed to initialize KITTI Object loader: {e}")
        
        # PKLot
        pklot_path = self.base_path / "pklot"
        if pklot_path.exists():
            try:
                self.loaders[DatasetType.PKLOT] = {
                    'train': PKLotLoader(str(pklot_path), 'train'),
                    'valid': PKLotLoader(str(pklot_path), 'valid'),
                    'test': PKLotLoader(str(pklot_path), 'test')
                }
                print("✅ PKLot loaders initialized")
            except Exception as e:
                print(f"❌ Failed to initialize PKLot loader: {e}")
        
        # Waymo
        waymo_path = self.base_path / "waymo" / "training"
        if waymo_path.exists():
            try:
                self.loaders[DatasetType.WAYMO] = WaymoLoader(str(waymo_path))
                print("✅ Waymo loader initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Waymo loader: {e}")
    
    def get_frame(self, dataset_type: DatasetType, idx: int, split: str = None) -> SensorFrame:
        """Get frame from specified dataset"""
        
        if dataset_type not in self.loaders:
            raise ValueError(f"Dataset {dataset_type} not available")
        
        loader = self.loaders[dataset_type]
        
        # Handle PKLot splits
        if dataset_type == DatasetType.PKLOT:
            if split is None:
                split = 'train'
            loader = loader[split]
        
        return loader.get_frame(idx)
    
    def get_dataset_info(self):
        """Get information about available datasets"""
        info = {}
        
        for dataset_type, loader in self.loaders.items():
            if dataset_type == DatasetType.PKLOT:
                info[dataset_type.value] = {
                    'train': loader['train'].num_samples,
                    'valid': loader['valid'].num_samples,
                    'test': loader['test'].num_samples
                }
            else:
                info[dataset_type.value] = {
                    'samples': loader.num_samples
                }
        
        return info

# Example usage and testing
if __name__ == "__main__":
    # Initialize the unified pipeline
    base_path = r"C:\Users\vnhar\Downloads\project1\extracted data"
    
    try:
        pipeline = UnifiedDataPipeline(base_path)
        
        # Print dataset information
        print("\n📊 Dataset Information:")
        info = pipeline.get_dataset_info()
        for dataset, stats in info.items():
            print(f"  {dataset}: {stats}")
        
        # Test loading a frame from each dataset
        print("\n🧪 Testing data loading:")
        
        # Test KITTI Object Detection
        if DatasetType.KITTI_OBJECT in pipeline.loaders:
            print("Loading KITTI frame...")
            kitti_frame = pipeline.get_frame(DatasetType.KITTI_OBJECT, 0)
            print(f"  ✅ KITTI frame {kitti_frame.frame_id}: "
                  f"Image shape: {kitti_frame.camera_data.left_image.shape}, "
                  f"LiDAR points: {len(kitti_frame.lidar_data.points)}, "
                  f"Labels: {len(kitti_frame.labels)}")
        
        # Test PKLot
        if DatasetType.PKLOT in pipeline.loaders:
            print("Loading PKLot frame...")
            pklot_frame = pipeline.get_frame(DatasetType.PKLOT, 0, 'train')
            print(f"  ✅ PKLot frame {pklot_frame.frame_id}: "
                  f"Image shape: {pklot_frame.camera_data.left_image.shape}")
        
        print("\n🎉 Data pipeline initialization successful!")
        
    except Exception as e:
        print(f"❌ Error initializing data pipeline: {e}")
        import traceback
        traceback.print_exc()