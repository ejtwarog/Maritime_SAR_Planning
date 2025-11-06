# Maritime_SAR_Planning

## Data

### NetCDF File Variables

The project uses NetCDF files containing oceanographic data from FVCOM (Finite Volume Coastal Ocean Model).

#### Current Velocity Variables
- **`u`**: Eastward water velocity (meters/second)
  - Range: -1.70 to 1.39 m/s
  - Dimensions: (time, siglay, nele)
  
- **`v`**: Northward water velocity (meters/second)
  - Range: -1.20 to 1.34 m/s
  - Dimensions: (time, siglay, nele)

**Derived Quantities:**
- Current speed: √(u² + v²)
- Current direction: atan2(v, u) (in radians)

#### Dimensions
- **time**: Time steps
- **siglay**: Sigma layers (vertical levels, 20 in sample data)
- **nele**: Grid elements/cells (102,264 in sample data)