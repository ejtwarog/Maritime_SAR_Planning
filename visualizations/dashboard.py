import xarray as xr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_data(nc_file):
    """Load NetCDF data and extract surface currents."""
    ds = xr.open_dataset(nc_file)
    
    # Check if this is a stations file or fields file
    if 'stations' in nc_file or 'lon' in ds.data_vars:
        # Stations file format
        u_surface = ds['u'].isel(siglay=0).values if 'siglay' in ds.dims else ds['u'].values
        v_surface = ds['v'].isel(siglay=0).values if 'siglay' in ds.dims else ds['v'].values
        lon = ds['lon'].values
        lat = ds['lat'].values
        time_var = ds['Times'].values if 'Times' in ds else ds['time'].values
    else:
        # Fields file format
        u_surface = ds['u'].isel(siglay=0).values  # (time, nele)
        v_surface = ds['v'].isel(siglay=0).values  # (time, nele)
        lon = ds['lonc'].values  # Element centers longitude
        lat = ds['latc'].values  # Element centers latitude
        time_var = ds['Times'].values
    
    # Convert longitude from 0-360 to -180-180 format
    lon = np.where(lon > 180, lon - 360, lon)
    
    # Calculate current speed and direction
    speed = np.sqrt(u_surface**2 + v_surface**2)
    direction = np.arctan2(v_surface, u_surface)
    
    return {
        'u': u_surface,
        'v': v_surface,
        'speed': speed,
        'direction': direction,
        'lon': lon,
        'lat': lat,
        'time': time_var,
        'ds': ds
    }

def create_quiver_plot(data, time_idx=0):
    """Create a quiver plot for currents at a specific time."""
    u = data['u'][time_idx]
    v = data['v'][time_idx]
    speed = data['speed'][time_idx]
    lon = data['lon']
    lat = data['lat']
    
    # Subsample for clarity (show every nth vector)
    subsample = 50
    indices = np.arange(0, len(lon), subsample)
    
    lon_sub = lon[indices]
    lat_sub = lat[indices]
    u_sub = u[indices]
    v_sub = v[indices]
    speed_sub = speed[indices]
    
    # Create quiver plot using scatter with arrows
    fig = go.Figure()
    
    # Add scatter plot colored by speed
    fig.add_trace(go.Scatter(
        x=lon_sub,
        y=lat_sub,
        mode='markers',
        marker=dict(
            size=8,
            color=speed_sub,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Speed (m/s)"),
            line=dict(width=0)
        ),
        text=[f"Speed: {s:.3f} m/s" for s in speed_sub],
        hoverinfo='text',
        name='Current Speed'
    ))
    
    # Add arrows using quiver-like representation
    arrow_scale = 0.005
    for i in range(len(lon_sub)):
        fig.add_annotation(
            x=lon_sub[i],
            y=lat_sub[i],
            ax=lon_sub[i] - u_sub[i] * arrow_scale,
            ay=lat_sub[i] - v_sub[i] * arrow_scale,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='rgba(0, 0, 0, 0.4)',
            showarrow=True
        )
    
    fig.update_layout(
        title=f"Surface Currents - Time: {data['time'][time_idx]}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        hovermode='closest',
        width=1000,
        height=800,
        showlegend=True
    )
    
    return fig


def create_slider_visualization(nc_file, output_file='surface_currents.html'):
    """Create an interactive time-slider visualization of surface currents on a map."""
    data = load_data(nc_file)
    
    n_times = data['u'].shape[0]
    
    # Create figure with slider and geographic projection
    fig = go.Figure()
    
    # Use all stations
    lon_sub = data['lon']
    lat_sub = data['lat']
    
    # Calculate bounds for map centering
    center_lon = np.mean(lon_sub)
    center_lat = np.mean(lat_sub)
    
    # Add traces for each time step
    for t in range(n_times):
        speed = data['speed'][t]
        speed_sub = speed
        
        # Add scatter trace on mapbox
        fig.add_trace(go.Scattermapbox(
            lon=lon_sub,
            lat=lat_sub,
            mode='markers',
            marker=dict(
                size=8,
                color=speed_sub,
                colorscale='Viridis',
                showscale=True,  # Show colorbar for all traces
                colorbar=dict(
                    title="Speed (m/s)",
                    thickness=15,
                    len=0.7,
                    x=1.02,
                    xanchor='left',
                    y=0.5,
                    yanchor='middle'
                ),
                cmin=np.min(data['speed']),
                cmax=np.max(data['speed']),
                opacity=0.7
            ),
            text=[f"Speed: {s:.3f} m/s" for s in speed_sub],
            hoverinfo='text',
            name=f'Time {t}',
            visible=(t == 0),  # Only first trace visible initially
            showlegend=False
        ))
    
    # Create slider steps with formatted time labels
    steps = []
    for t in range(n_times):
        # Build visibility list for traces only
        trace_visibility = [i == t for i in range(n_times)]
        
        # Format time label - convert to datetime if needed
        time_val = data['time'][t]
        if isinstance(time_val, bytes):
            time_str = time_val.decode('utf-8')
        else:
            time_str = str(time_val)
        
        # Extract just the date and time portion (first 16 characters: YYYY-MM-DDTHH:MM)
        time_label = time_str[:16].replace('T', ' ')
        
        step = dict(
            method="update",
            args=[{"visible": trace_visibility},
                  {"title": f"Surface Currents - {time_label}"}],
            label=time_label
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Time: "},
        pad={"b": 10, "t": 50},
        len=0.9,
        x=0.05,
        y=0,
        steps=steps
    )]
    
    # Format initial time label
    time_val = data['time'][0]
    if isinstance(time_val, bytes):
        time_str = time_val.decode('utf-8')
    else:
        time_str = str(time_val)
    time_label = time_str[:16].replace('T', ' ')
    
    fig.update_layout(
        title=dict(
            text=f"Surface Currents - {time_label}",
            font=dict(size=18, color='#00d4ff', family='Segoe UI, sans-serif', weight='normal'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        hovermode='closest',
        autosize=True,
        margin=dict(l=0, r=120, t=80, b=120),
        sliders=sliders,
        showlegend=False,
        paper_bgcolor='#0a1428',
        plot_bgcolor='#0a1428',
        font=dict(color='#ffffff', family='Segoe UI, sans-serif', size=11),
        mapbox=dict(
            style='carto-darkmatter',
            center=dict(lon=center_lon, lat=center_lat),
            zoom=9
        )
    )
    
    # Update slider styling with step buttons
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="<b>Time:</b> ",
                font=dict(size=13, color='#00d4ff'),
                visible=True,
                xanchor='center'
            ),
            pad=dict(b=15, t=50),
            len=0.75,
            x=0.15,
            y=0,
            steps=sliders[0]['steps'],
            font=dict(size=10, color='#ffffff'),
            transition=dict(duration=300)
        )]
    )
    
    # Add dashboard title
    dashboard_title_html = '''
    <div id="dashboard-title" style="
        position: fixed;
        top: 15px;
        left: 20px;
        z-index: 1000;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    ">
        <h1 style="
            margin: 0;
            padding: 0;
            font-size: 28px;
            font-weight: 300;
            color: #ffffff;
            letter-spacing: 0.5px;
        ">Maritime Search and Rescue</h1>
        <p style="
            margin: 2px 0 0 0;
            padding: 0;
            font-size: 14px;
            color: #00d4ff;
            font-weight: 300;
            letter-spacing: 0.3px;
        ">Algorithms for Decision-Making</p>
    </div>
    '''
    
    # No custom buttons - keep it clean
    step_buttons_html = ''
    
    # Add full-screen CSS
    fig.write_html(
        output_file,
        config={'responsive': True, 'displayModeBar': True}
    )
    
    # Inject CSS for full-screen styling and dark theme
    with open(output_file, 'r') as f:
        html_content = f.read()
    
    # Add full-screen CSS with dark navy theme
    full_screen_css = '''
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        html, body {
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #0a1428;
        }
        .plotly-graph-div {
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .modebar {
            background-color: #1a3a52 !important;
            border-radius: 5px;
            z-index: 100 !important;
        }
        .modebar-btn {
            color: #ffffff !important;
        }
        .modebar-btn:hover {
            background-color: #2a5a7f !important;
        }
        .slice {
            background-color: #0a1428;
        }
        /* Ensure slider is visible */
        .slider {
            z-index: 50 !important;
            background-color: #1a3a52 !important;
            border-top: 1px solid #2a5a7f !important;
            visibility: visible !important;
        }
        /* Ensure colorbar is visible */
        .colorbar {
            background-color: #0a1428 !important;
            visibility: visible !important;
        }
        .xtick, .ytick {
            color: #ffffff !important;
        }
        /* Ensure all text is visible */
        text {
            fill: #ffffff !important;
        }
    </style>
    '''
    
    html_content = html_content.replace('</head>', full_screen_css + '</head>')
    html_content = html_content.replace('</body>', dashboard_title_html + step_buttons_html + '</body>')
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization saved to {output_file}")
    
    return fig


if __name__ == "__main__":
    nc_file = "data/sfbofs.t15z.20251105.stations.forecast.nc"
    
    # Create and save the interactive visualization
    fig = create_slider_visualization(nc_file)
    fig.show()
