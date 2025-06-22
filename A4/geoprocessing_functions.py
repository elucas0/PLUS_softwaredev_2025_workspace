"""Geoprocessing Functions for Satellite Data
    
This module provides functions to load satellite data from TIFF files, extract geospatial metadata,
and preprocess the data for various machine learning models, it supports reading specific bands, 
resizing images, and normalizing pixel values :

    * load_satellite_data - reads a TIFF (.tif) file and returns the data as a PyTorch tensor along with metadata.
    * extract_geo_metadata - extracts geospatial metadata from the loaded data.
    * preprocess_for_models - preprocesses the data for different models, handling band counts.
"""

import torch
import rasterio
import torchvision.transforms as transforms
from typing import Dict, Any, Optional, List

def load_satellite_data(file_path: str, 
                       bands: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Load satellite data from TIFF file.
    
    Args:
        file_path: Path to TIFF file
        bands: List of band indices to load (1-indexed). If None, loads all bands
    
    Returns:
        Dict containing:
            - 'data': torch.Tensor of shape (C, H, W)
            - 'transform': affine transform matrix
            - 'crs': coordinate reference system
            - 'bounds': spatial bounds (minx, miny, maxx, maxy)
    """

    with rasterio.open(file_path) as dataset:
        # Determine bands to read
        if bands is None:
            bands = list(range(1, dataset.count + 1))
        
        data = dataset.read(bands)
        
        # Convert to tensor
        data_tensor = torch.from_numpy(data).float()
        
        return {
            'data': data_tensor,
            'transform': dataset.transform,
            'crs': dataset.crs,
            'bounds': dataset.bounds
        }

def extract_geo_metadata(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract geospatial metadata from loaded satellite data.
    
    Args:
        image_data: Dictionary from load_satellite_data function
    
    Returns:
        Dictionary containing geospatial metadata
    """
    bounds = image_data['bounds']
    transform = image_data['transform']
    data_shape = image_data['data'].shape
    
    metadata = {
        'crs': str(image_data['crs']),
        'bounds': bounds,
        'shape': data_shape,  # (channels, height, width)
        'pixel_size_x': abs(transform.a),
        'pixel_size_y': abs(transform.e),
        'width_meters': abs(bounds[2] - bounds[0]),
        'height_meters': abs(bounds[3] - bounds[1]),
        'area_km2': (abs(bounds[2] - bounds[0]) * abs(bounds[3] - bounds[1])) / 1_000_000,
        'center_lon': (bounds[0] + bounds[2]) / 2,
        'center_lat': (bounds[1] + bounds[3]) / 2
    }
    
    return metadata

def preprocess_for_models(image_data: Dict[str, Any], 
                         target_models: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Preprocess satellite data for different models.
    
    Args:
        image_data: Dictionary from load_satellite_data function
        target_models: Dict with model names as keys and specs as values
                      Each spec should contain:
                      - 'input_size': (height, width) tuple or int
                      - 'bands': number of expected input bands
                      - 'normalize': dict with 'mean' and 'std' lists (optional)
    
    Returns:
        Dict with model names as keys and preprocessed tensors as values
    """
    data_tensor = image_data['data']  # Shape: (C, H, W)
    preprocessed = {}
    
    for model_name, specs in target_models.items():
        processed_tensor = data_tensor.clone()
        
        # Handle band count
        expected_bands = specs.get('bands', processed_tensor.shape[0])
        current_bands = processed_tensor.shape[0]
        
        if current_bands != expected_bands:
            if current_bands > expected_bands:
                # Take first N bands
                processed_tensor = processed_tensor[:expected_bands]
            elif current_bands < expected_bands:
                if current_bands == 1:
                    # Repeat single band (grayscale to RGB)
                    processed_tensor = processed_tensor.repeat(expected_bands, 1, 1)
                else:
                    # Pad with zeros
                    padding_bands = expected_bands - current_bands
                    padding = torch.zeros(padding_bands, *processed_tensor.shape[1:])
                    processed_tensor = torch.cat([processed_tensor, padding], dim=0)
        
        # Handle spatial resizing
        input_size = specs.get('input_size', processed_tensor.shape[1:])
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        
        current_size = (processed_tensor.shape[1], processed_tensor.shape[2])
        if current_size != input_size:
            resize_transform = transforms.Resize(input_size)
            processed_tensor = resize_transform(processed_tensor)
        
        # Apply normalization if specified
        if 'normalize' in specs and specs['normalize'] is not None:
            normalize_params = specs['normalize']
            mean = torch.tensor(normalize_params['mean']).view(-1, 1, 1)
            std = torch.tensor(normalize_params['std']).view(-1, 1, 1)
            processed_tensor = (processed_tensor - mean) / std
        
        preprocessed[model_name] = processed_tensor
    
    return preprocessed
