import xarray as xr
import os

# Path to the NetCDF file
nc_file = "data/sample_sfbofs.t03z.20251023.fields.f001.nc"

# Open the NetCDF file
ds = xr.open_dataset(nc_file)

print("=" * 80)
print("NetCDF FILE STRUCTURE")
print("=" * 80)
print(ds)

print("\n" + "=" * 80)
print("DIMENSIONS")
print("=" * 80)
for dim, size in ds.dims.items():
    print(f"  {dim}: {size}")

print("\n" + "=" * 80)
print("VARIABLES")
print("=" * 80)
for var_name in ds.data_vars:
    var = ds[var_name]
    print(f"\n  {var_name}")
    print(f"    Shape: {var.shape}")
    print(f"    Dtype: {var.dtype}")
    print(f"    Dimensions: {var.dims}")
    if var.attrs:
        print(f"    Attributes:")
        for attr_key, attr_val in var.attrs.items():
            print(f"      {attr_key}: {attr_val}")

print("\n" + "=" * 80)
print("COORDINATES")
print("=" * 80)
for coord_name in ds.coords:
    coord = ds[coord_name]
    print(f"\n  {coord_name}")
    print(f"    Shape: {coord.shape}")
    print(f"    Dtype: {coord.dtype}")
    if coord.attrs:
        print(f"    Attributes:")
        for attr_key, attr_val in coord.attrs.items():
            print(f"      {attr_key}: {attr_val}")

print("\n" + "=" * 80)
print("GLOBAL ATTRIBUTES")
print("=" * 80)
if ds.attrs:
    for attr_key, attr_val in ds.attrs.items():
        print(f"  {attr_key}: {attr_val}")
else:
    print("  (No global attributes)")

ds.close()
