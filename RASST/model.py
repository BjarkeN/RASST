import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy

class model():
    
    
    
    def __init__(self, altimetry, mask, dem):
        
        self.altimetry = altimetry
        self.mask = mask
        self.dem = dem
        
    def sample_elevations(self, sampling="ang_rect"):
        
        # Determine sampling grid
        grid = self.sampling_grid(self.theta, height=300, width=100)
        
        # Determine footprint centerline
        line_loc = np.copy(self.centerline).T
        line_loc = np.flip(line_loc,1)
        line_loc = line_loc.tolist()

        # =======================================================
        # Sample from dem
        
        # Convert from centerline latlon to dem index
        line_index = latlon2idx(self.dem, line_loc)

        # Create line over dem
        N_samplepoints = 4000
        X = np.linspace(line_index[0,1], line_index[1,1], N_samplepoints).astype(int)
        Y = np.linspace(line_index[0,0], line_index[1,0], N_samplepoints).astype(int)
        x = np.linspace(line_loc[0][1], line_loc[0][0], N_samplepoints)
        y = np.linspace(line_loc[1][1], line_loc[1][0], N_samplepoints)

        # Extract the values along the line
        elevations = []
        xi = []
        yi = []
        sampling_method = sampling # <------- VARIABLE [ang_rect / square]
        for i in np.arange(0,N_samplepoints):
            match sampling_method:
                case "square":
                    #==============
                    # Square sampling grid    
                    slice_size = 10  
                    elevations.append(np.nanmean(self.dem.data.values[(X[i]-slice_size):(X[i]+slice_size),
                                                              (Y[i]-slice_size):(Y[i]+slice_size)]))
                    # =============
                case "ang_rect":
                    # =============
                    # Directional rectangular sampling grid
                    val = self.dem.data.values[grid[0]+X[i],
                                               grid[1]+Y[i]]
                    if np.all( np.isnan( val ) ): # if there is only nans
                        elevations.append(np.nan)
                    else:
                        elevations.append(np.nanmean(val))
                    # =============
            # Save corresponding lat lon
            xi.append(x[i])
            yi.append(y[i])
            
        # =======================================================
        # Sample from mask
        
        # Convert from centerline latlon to water mask index
        mask_index = latlon2idx(self.mask, line_loc)
        
        # Create line over water mask
        X = np.linspace(mask_index[0,0], mask_index[1,0], N_samplepoints).astype(int)
        Y = np.linspace(mask_index[0,1], mask_index[1,1], N_samplepoints).astype(int)
        
        # Extract the values along the line
        surface_flags = []
        mask_idx = self.mask.data.T
        for i in np.arange(0,N_samplepoints):
            match sampling_method:
                case "square":
                    #==============
                    # Square sampling grid    
                    slice_size = 10  
                    mask_vals = mask_idx[(X[i]-slice_size):(X[i]+slice_size),
                                         (Y[i]-slice_size):(Y[i]+slice_size)]
                    vals, counts = np.unique(mask_vals, return_counts=True)
                    surface_flags.append(vals[np.argmax(counts)])
                    # =============
                case "ang_rect":
                    # =============
                    # Directional rectangular sampling grid
                    val = mask_idx[grid[0]+X[i],
                                   grid[1]+Y[i]]
                    if np.all( np.isnan( val ) ): # if there is only nans
                        surface_flags.append(np.nan)
                    else:
                        surface_flags.append(np.nanmean(val))
                # =============
        
        # Convert extracted data to numpy arrays
        elevations = np.array(elevations)
        xi = np.array(xi)
        yi = np.array(yi)
        surface_flags = np.array(surface_flags)
        #vi = np.array(vi)
        
        # Save internally
        self.elevations = elevations
        self.surface_flags = surface_flags
        self.xi = xi
        self.yi = yi
        
        
        # Split data in water and land
        #mi[mi<0.5] = np.nan
        #vi[vi<0.5] = np.nan
        #mi[mi>=0.5] = zi[mi>=0.5]
        #vi[vi>=0.5] = zi[vi>=0.5]
        #zi[np.logical_or(~np.isnan(mi),~np.isnan(vi))] = np.nan
        
        return elevations, surface_flags
    #def sample_synth_waveform(self):
        
    def sampling_grid(self, theta, height=300, width=100):
        
        # Determine sampling grid
        size_of_px = self.dem.px_size
        alon_grid_size = np.ceil(height/size_of_px)
        cross_grid_size = np.ceil(width/size_of_px)
        grid = sampling_grid(cross_grid_size, alon_grid_size, theta)
        return grid

    def footprint(self, loc_id, width=12000, height=300):
        
        # Determine footprint
        self.location = np.array([ self.altimetry.data["longitude"][loc_id],
                                   self.altimetry.data["latitude"][loc_id] ])
        self.location_next = np.array([ self.altimetry.data["longitude"][loc_id+1],
                                        self.altimetry.data["latitude"][loc_id+1] ])
        self.footprint, self.theta, self.aspect = determine_footprint(width, height,
                                                       self.location, self.location_next)
        self.centerline = np.array([np.mean(self.footprint[:,[0,3]],1),
                                    np.mean(self.footprint[:,[1,2]],1)])
        
        """assert ( np.min(self.footprint[0,:]) > np.min(self.dem.extent_geo[0,:]) and 
                 np.max(self.footprint[0,:]) < np.max(self.dem.extent_geo[0,:]) and 
                 np.min(self.footprint[1,:]) > np.min(self.dem.extent_geo[1,:]) and 
                 np.max(self.footprint[1,:]) < np.max(self.dem.extent_geo[1,:]) ), "Footprint exceeded DEM size"
        """
        return self.footprint, self.centerline, self.theta, self.aspect

def determine_footprint(width, height, center, next_point):
    '''Function to determine the rotated footprint of a satellite
    
    Inputs:
        (float)width: Width of the footprint unit: meters
        (float)height: Height of the footprint unit: meters
        (2d numpy array)center: np.array([longitude, latitude]) of satellite
            in WGS84 decimal degrees (floats)
        (1x2 numpy array)center: np.array([longitude, latitude]) of satellite
            in WGS84 decimal degrees (floats), for the position next in time
            used for obtaining the angle of rotation
        
    Output:
        (2x5 numpy array)X: numpy array of decimal degrees of the four corners,
            and the first to complete the loop, of the footprint of the satellite.
    
    Example: 
    width = 5000
    height = 300
    centerpoint = 12
    center = np.array([dat["longitude"][centerpoint], dat["latitude"][centerpoint]])
    next_point = np.array([dat["longitude"][centerpoint+1], dat["latitude"][centerpoint+1]])
    footprint = determine_footprint(5000, 300, center, next_point)
    
    '''
    # Scale factors from degrees to meters
    fac_lat = 111320
    fac_lon = 40075000 * np.cos(center[1] / 180 * np.pi) / 360
    
    # Determine the non-rotated footprint
    footprint_lon = center[0] + np.array([-1,1,1,-1,-1]) * width / 2 / fac_lon
    footprint_lat = center[1] + np.array([-1,-1,1,1,-1]) * height / 2 / fac_lat
    aspect_ratio = fac_lon / fac_lat
    aspect = [fac_lat, fac_lon, aspect_ratio]

    # Use rotation matrix to rotate the centered footprint in meters
    dif_lon = center[0] - next_point[0]
    dif_lat = center[1] - next_point[1]
    theta = -np.arctan( ( dif_lon*fac_lon ) / ( dif_lat*fac_lat ) )
    mu = np.array([np.array([center[0]*fac_lon, center[1]*fac_lat])]).T
    X = np.array([footprint_lon*fac_lon, footprint_lat*fac_lat]) - mu
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X_rot = ((rot @ X) + mu) / np.array([[fac_lon, fac_lat]]).T
    
    return X_rot, theta, aspect
def latlon2idx(im, poly):
    poly_coord = np.flip(np.vstack((poly)).T)
    # Get ids from lat lon
    ids = []
    for val in poly_coord:
        id_y = np.abs(im.y - val[0]).argmin()
        id_x = np.abs(im.x - val[1]).argmin()
        ids.append([id_x, id_y])
    line_index = np.array(ids)
    return line_index
# Determine sampling grid
def sampling_grid(width, height, angle):
    theta = -angle
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    grid = np.array(np.meshgrid(np.arange(-np.round(width/2),np.round(width/2)),
                                np.arange(-np.round(height/2),np.round(height/2))))
    grid = np.array([grid[0].reshape(-1,1), grid[1].reshape(-1,1)]).T
    grid = grid[0].T
    grid = rot @ grid
    grid = np.round(grid).astype(int)
    return grid
def downsampling(im_in, gridsize):
    im = np.copy(im_in)
    downsample_grid = np.array(np.meshgrid(np.linspace(0,im.shape[1]-1, gridsize).astype(int),
                                    np.linspace(0,im.shape[0]-1, gridsize).astype(int)))
    downsample_grid = np.array([downsample_grid[0].reshape(-1,1), downsample_grid[1].reshape(-1,1)])
    im = im[downsample_grid[1], downsample_grid[0]].reshape(gridsize,gridsize)
    return im