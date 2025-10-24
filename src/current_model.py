import pandas as pd
import numpy as np
import xarray as xr

# Note this file will host functions to help with loading and manipulating current models. 
# For now, we will focus on loading a NetCDF current model and extracting some basic metadata.
# Note: We will try to transition to also using GRIB files in the future.

def load_current_model(model_path: str):
    """
    Load the current model (NetCDF format) from the specified path.
    """
    model = xr.open_dataset(model_path)
    return model

def get_metadata(model):
    """
    Extract metadata from the current model.
    """
    metadata = Dict()
    metadata["name"] = model.attrs["name"]
    metadata["start_time"] = model.attrs["start_time"]
    metadata["end_time"] = model.attrs["end_time"]
    return metadata
