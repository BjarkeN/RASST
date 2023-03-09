import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy

class model():
    
    
    
    def __init__(self, altimetry, mask, dem, location):
        
        self.altimetry = altimetry
        self.mask = mask
        self.dem = dem
        self.location = location
        
    def sample_elevations(self, sampling="ang_rect", N_samplepoints=4000):
        """_summary_

        Args:
            sampling (str, optional): _description_. Defaults to "ang_rect".

        Returns:
            _type_: _description_
        """
        
        
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
        #N_samplepoints = 4000
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
                    surface_flags.append(np.round(vals[np.argmax(counts)]))
                    # =============
                case "ang_rect":
                    # =============
                    # Directional rectangular sampling grid
                    mask_vals = mask_idx[grid[0]+X[i],
                                         grid[1]+Y[i]]
                    if np.all( np.isnan( val ) ): # if there is only nans
                        surface_flags.append(np.nan)
                    else:
                        vals, counts = np.unique(mask_vals, return_counts=True)
                        surface_flags.append(np.round(vals[np.argmax(counts)]))
                # =============
        
        # Convert extracted data to numpy arrays
        elevations = np.array(elevations)
        x = np.array(xi)
        y = np.array(yi)
        surface_flags = np.array(surface_flags)
        
        # Create reflectivity factor
        self.reflectance = np.ones(np.unique(surface_flags).shape[0])
        
        # Determine across-track distance
        x_diff = np.diff(x) * self.aspect[1]
        y_diff = np.diff(y) * self.aspect[0]
        along = np.r_[np.zeros(1), np.cumsum(np.sqrt(x_diff**2 + y_diff**2))]
        along_centered = along - np.median(along)
        
        # Save internally
        self.elevations = elevations
        self.surface_flags = surface_flags
        self.x = x
        self.y = y
        self.along_centered = along_centered
        
        return elevations, surface_flags, along_centered
    
    def show_sampling(self, background, margin=0.05):
        
        # Test line
        fig, ax = plt.subplots(figsize=(12,7))

        # Background
        plot_img = background.get_plot_array()
        ax.imshow(plot_img, extent=background.extent_geo)

        # Plot sat path
        ax.plot(self.x, self.y, 'm--')
        ax.plot(self.footprint[0], self.footprint[1], 'm-')
        ax.plot(self.altimetry.data["longitude"][:],
                self.altimetry.data["latitude"][:], 'ro-', markersize=2, lw=0.5)


        v_factor = 5e-4
        v_offset = np.nanmin(self.elevations)
        #ax.vlines(self.x, self.y+(self.elevations-v_offset)*v_factor, self.y, lw=0.5, color='b',alpha=0.05)
        #ax.plot(self.x, self.y+(self.elevations-v_offset)*v_factor, 'b')
        if np.unique(self.surface_flags).shape[0] == 3:
            cols = ["y","b","g"]
            for f in np.unique(self.surface_flags):
                f = int(f)
                aoi = (self.surface_flags == f)
                ax.vlines(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor, self.y[aoi], lw=0.5, color=cols[f],alpha=0.05)
                ax.scatter(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor,1, cols[f])
        else:
            for f in np.unique(self.surface_flags):
                f = int(f)
                aoi = (self.surface_flags == f)
                ax.vlines(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor, self.y[aoi], lw=0.5,alpha=0.05)
                ax.scatter(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor,1)
            
        #ax.vlines(xi, yi+(vi-v_offset)*v_factor, yi, lw=0.5, color='g',alpha=0.05)
        #ax.plot(xi, yi+(vi-v_offset)*v_factor, 'g')
        #ax.vlines(xi, yi+(zi-v_offset)*v_factor, yi, lw=0.5, color='y',alpha=0.05)
        #ax.plot(xi, yi+(zi-v_offset)*v_factor, 'y')

        legend = ["DEM Extent","Footprint Centerline", "Footprint Outline", "Satellite Track", "Water Flag", "DEM Elevation"]
        ax.legend(legend, loc="upper left")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")

        ax.set_xlim([self.footprint[0,:].min()-margin, self.footprint[0,:].max()+margin])
        ax.set_ylim([self.footprint[1,:].min()-margin, self.footprint[1,:].max()+margin])

        plt.show()
    
    def synthetic_waveform(self, elevations=None, flags=None, along=None, reflectance=None,
                           show_data=False, output="numpy"):
        """Forward Model for power waveform given a specific surface elevation

        Args:
            elevations (_type_, optional): _description_. Defaults to None.
            flags (_type_, optional): _description_. Defaults to None.
            along (_type_, optional): _description_. Defaults to None.
            show_data (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        
        # Update values if provided
        if np.any(elevations != None):
            self.elevations = elevations
        if np.any(flags != None):
            self.surface_flags = flags
        if np.any(reflectance != None):
            self.reflectance = reflectance
        if np.any(along != None):
            self.along_centered = along
        
        # Determine wavefront shape
        sat_altitude = self.altimetry.data["altitude"][self.location]
        #if output == "torch":
        #    along = self.along_centered.detach().numpy()
        #    wavefront_height = wavefront(sat_altitude, along)
        #else:
        #    wavefront_height = wavefront(sat_altitude, self.along_centered)
        wavefront_height = wavefront(sat_altitude, self.along_centered)
        
        # Smooth the surface
        smooth_n = 5
        #smooth_zi = np.convolve(np.ones(smooth_n), self.elevations, mode="same")/smooth_n # Smooth
        #idxs = [np.arange(-int(smooth_n/2),int(smooth_n/2))]
        #smooth_zi[idxs] = np.nan
        if output == "torch":
            ma = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=smooth_n,
                        stride=1, padding="same", bias=False)
            ma.weight.data = torch.ones(ma.weight.data.size())
            smooth_zi = ma(self.elevations[None,:])/smooth_n
            smooth_zi = smooth_zi[0]
        else:
            smooth_zi = self.elevations

        # Show data
        if show_data == True:
            fig, ax = plt.subplots(1,figsize=(10,4))
            if np.unique(self.surface_flags).shape[0] == 3:
                cols = ["y","b","g"]
                for f in np.unique(self.surface_flags):
                    f = int(f)
                    aoi = (self.surface_flags == f)
                    ax.scatter(self.along_centered[aoi], smooth_zi[aoi], 1, cols[f])
                
                v_offset = np.nanmin(self.elevations)
                ax.plot(self.along_centered, wavefront_height-2+v_offset, 'r')
                ax.plot(self.along_centered, wavefront_height+35+v_offset, 'r')
                #ax.plot(self.along_centered, wavefront_height+58+v_offset, 'r')

                ax.legend(["Land","Water","Vegetation","Wavefront Illustration"])
            else:
                for f in np.unique(self.surface_flags):
                    f = int(f)
                    aoi = (self.surface_flags == f)
                    ax.scatter(self.along_centered[aoi], smooth_zi[aoi], 1)
                v_offset = np.nanmin(self.elevations)
                ax.plot(self.along_centered, wavefront_height-2+v_offset, 'r')
                ax.plot(self.along_centered, wavefront_height+35+v_offset, 'r')
            ax.set_xlabel("Crosstrack distance [m]", fontweight="bold")
            ax.set_ylabel("Height [m]", fontweight="bold")
            plt.show()

        # Correct for wavefront
        if output == "torch":
            range_corrected = smooth_zi - torch.from_numpy(wavefront_height)
        else:
            range_corrected = smooth_zi - wavefront_height
        
        # Determine range
        ranges = self.altimetry.data["heights"][self.location]

        N = ranges.shape[0]
        synth_ranges = ranges
        
        if output == "torch":
            envelope = torch.zeros(N)
            synth_rangepower = torch.zeros((np.unique(self.surface_flags).shape[0], N))
        else:
            envelope = np.zeros(N)
            synth_rangepower = np.zeros((np.unique(self.surface_flags).shape[0], N))
            
        f_idx = 0
        for f in np.unique(self.surface_flags):
            # Extract the ranges corresponding to the current flag
            range_corrected_z = range_corrected[self.surface_flags==f]
            along_centered_z = self.along_centered[self.surface_flags==f]
            
            for i in range(N-1):
                count_z = np.logical_and(range_corrected_z < synth_ranges[i],
                                        range_corrected_z > synth_ranges[i+1])
                along_dists_z = along_centered_z[np.logical_and(range_corrected_z < synth_ranges[i],
                                                                range_corrected_z > synth_ranges[i+1])]
                # Scale with distance from center
                scale_param = 0.09#0.03 # lower number means more weight to tails
                synth_rangepower[f_idx,i] = count_z.sum()*self.reflectance[f_idx] \
                                                                * illumination(along_dists_z, scale_param, 
                                                                              mode="normal") if np.any(along_dists_z) else 0
            
            smooth_n = 3
            #synth_rangepower[f_idx,:] = np.convolve(np.ones(smooth_n), synth_rangepower[int(f),:], mode="same")/smooth_n # Smooth
                        
            # Add individual contribution to the full waveform
            envelope += synth_rangepower[f_idx,:]
            
            f_idx += 1
        # Normalize the individual contributions
        f_idx = 0
        for f in np.unique(self.surface_flags):
            synth_rangepower[f_idx,:] = synth_rangepower[f_idx,:]/envelope.max()
            f_idx += 1
            
        # Smooth the envelope
        n_smooth = 3
        if output == "torch":
            kernel = torch.ones(n_smooth)
            kernel = kernel[None,None,:]
            # Apply smoothing
            envelope = (torch.conv1d(envelope[None,None,:], kernel)/n_smooth)[0,0,:]
        else:
            envelope = np.convolve(np.ones(n_smooth), envelope, mode="same")/n_smooth
            
        # Normalize the envelope
        envelope = envelope/envelope.max()
        
        match output:
            case "numpy":
                return envelope, synth_rangepower, ranges
            case "torch":
                return envelope[:256]
        
    def sampling_grid(self, theta, height=300, width=100):
        
        # Determine sampling grid
        size_of_px = self.dem.px_size
        alon_grid_size = np.ceil(height/size_of_px)
        cross_grid_size = np.ceil(width/size_of_px)
        grid = sampling_grid(cross_grid_size, alon_grid_size, theta)
        return grid

    def footprint(self, width=12000, height=300):
        
        # Determine footprint
        self.first = np.array([ self.altimetry.data["longitude"][self.location],
                                self.altimetry.data["latitude"][self.location] ])
        self.next = np.array([ self.altimetry.data["longitude"][self.location+1],
                               self.altimetry.data["latitude"][self.location+1] ])
        self.footprint, self.theta, self.aspect = determine_footprint(width, height,
                                                       self.first, self.next)
        self.centerline = np.array([np.mean(self.footprint[:,[0,3]],1),
                                    np.mean(self.footprint[:,[1,2]],1)])
        
        """assert ( np.min(self.footprint[0,:]) > np.min(self.dem.extent_geo[0,:]) and 
                 np.max(self.footprint[0,:]) < np.max(self.dem.extent_geo[0,:]) and 
                 np.min(self.footprint[1,:]) > np.min(self.dem.extent_geo[1,:]) and 
                 np.max(self.footprint[1,:]) < np.max(self.dem.extent_geo[1,:]) ), "Footprint exceeded DEM size"
        """
        return self.footprint, self.centerline, self.theta, self.aspect

# ==============================================================================
# FUNCTIONS

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

def wavefront(H, x):
    """Calculate the vertical difference from a straight line of a wavefront

    Args:
        H (float): Height of a satellite
        x (numpy array): distance cross track

    Returns:
        y (numpy array): height along track
    """
    return -(np.sqrt( H**2 - x**2 ) - H)

def illumination(x, scale_param, mode = "normal", output="numpy"):
    """_summary_

    Args:
        x (_type_): _description_
        scale_param (_type_): _description_
        mode (str, optional): _description_. Defaults to "normal".

    Returns:
        _type_: _description_
    """
    nu = 50
    if mode == "normal":
        if output == "torch":
            return torch.exp(-scale_param*(torch.mean(x)/(1e3))**2)**2
        elif output == "numpy":
            return np.exp(-scale_param*(np.mean(x)/(1e3))**2)**2
    elif mode == "sinh":
        return np.exp(-scale_param*(np.sinh( 0.9 * ( np.arcsinh(np.mean(x)/1e3)) + 0))**2)
    
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