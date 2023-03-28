import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy
from datetime import datetime
import json

# Libraries for bayesian inference
import torch
import pyro
from pyro.infer import MCMC, NUTS, Predictive
import numpyro
import jax.numpy as jnp
from jax import random

class model():
    
    
    
    def __init__(self, altimetry, mask, dem, location):
        
        self.altimetry = altimetry
        self.mask = mask
        self.dem = dem
        self.location = location
        
    def sample_elevations(self, sampling="ang_rect", N_samplepoints=4000, n_segments=10):
        """_summary_

        Args:
            sampling (str, optional): _description_. Defaults to "ang_rect".

        Returns:
            _type_: _description_
        """
        
        
        # Determine sampling grid
        grid = self.sampling_grid(self.theta, height=300, width=100)
        
        # Determine number of sampling id's
        self.n_sampling_ids = n_segments
        
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
        
        # Reset lowest number of surface flag
        surface_flags[surface_flags==0] += np.unique(surface_flags)[1]-1
        
        # Create reflectivity factor
        self.reflectance = np.ones(np.unique(surface_flags).shape[0])
        
        # Determine across-track distance
        x_diff = np.diff(x) * self.aspect[1]
        y_diff = np.diff(y) * self.aspect[0]
        along = np.r_[np.zeros(1), np.cumsum(np.sqrt(x_diff**2 + y_diff**2))]
        along_centered = along - np.median(along)
        
        # Pass segment id's
        segment_id = np.linspace(0, self.n_sampling_ids, N_samplepoints).astype(int)
        
        # Save internally
        self.elevations = elevations
        self.surface_flags = surface_flags
        self.x = x
        self.y = y
        self.along_centered = along_centered
        self.segment_id = segment_id
        
        return elevations, surface_flags, along_centered, segment_id
    
    def show_sampling(self, background, margin=0.05, colors="flags"):
        
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
        match colors:
            case "flags":
                for f in np.unique(self.surface_flags):
                    f = int(f)
                    aoi = (self.surface_flags == f)
                    ax.vlines(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor, self.y[aoi], lw=0.5,alpha=0.05)
                    ax.scatter(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor,1)
            case "segments":
                for f in np.unique(self.segment_id):
                    f = int(f)
                    aoi = (self.segment_id == f)
                    ax.vlines(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor, self.y[aoi], lw=0.5,alpha=0.05)
                    ax.scatter(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor,1)
            case "land_water_ground":
                cols = ["y","b","g"]
                for f in np.unique(self.surface_flags):
                    f = int(f)
                    aoi = (self.surface_flags == f)
                    ax.vlines(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor, self.y[aoi], lw=0.5, color=cols[f],alpha=0.05)
                    ax.scatter(self.x[aoi], self.y[aoi]+(self.elevations[aoi]-v_offset)*v_factor,1, cols[f])
            
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
        
    def show_profile(self, colors="flags"):
        # Determine wavefront shape
        sat_altitude = self.altimetry.data["altitude"][self.location]
        wavefront_height = wavefront(sat_altitude, self.along_centered)
        
        # Smooth the surface
        #smooth_n = 5
        #smooth_zi = np.convolve(np.ones(smooth_n), self.elevations, mode="same")/smooth_n
        smooth_zi = self.elevations

        fig, ax = plt.subplots(figsize=(10,4))
        
        match colors:
            case "flags":
                for f in np.unique(self.surface_flags):
                    f = int(f)
                    aoi = (self.surface_flags == f)
                    ax.scatter(self.along_centered[aoi], smooth_zi[aoi], 1)
                v_offset = np.nanmin(self.elevations)
                ax.plot(self.along_centered, wavefront_height-2+v_offset, 'k--')
                ax.plot(self.along_centered, wavefront_height+35+v_offset, 'k--', label='_nolegend_')
                ax.legend([ i for i in range(np.unique(self.surface_flags).shape[0])] + ["Altimeter Wavefront"])
            case "segments":
                for f in np.unique(self.segment_id):
                    f = int(f)
                    aoi = (self.segment_id == f)
                    ax.scatter(self.along_centered[aoi], smooth_zi[aoi], 1)
                v_offset = np.nanmin(self.elevations)
                ax.plot(self.along_centered, wavefront_height-2+v_offset, 'k--')
                ax.plot(self.along_centered, wavefront_height+35+v_offset, 'k--', label='_nolegend_')
                ax.legend([ i for i in range(np.unique(self.segment_id).shape[0])] + ["Wavefront"])
            case "land_water_vegetation":
                cols = ["y","b","g"]
                for f in np.unique(self.surface_flags):
                    f = int(f)
                    aoi = (self.surface_flags == f)
                    ax.scatter(self.along_centered[aoi], smooth_zi[aoi], 1, cols[f])
                v_offset = np.nanmin(self.elevations)
                ax.plot(self.along_centered, wavefront_height-2+v_offset, 'k--')
                ax.plot(self.along_centered, wavefront_height+35+v_offset, 'k--', label='_nolegend_')
                ax.legend(["Land","Water","Vegetation","Wavefront Illustration"])

        ax.set_xlabel("Crosstrack distance [m]", fontweight="bold")
        ax.set_ylabel("Height [m]", fontweight="bold")
        plt.show()
    
    def synthetic_waveform(self, elevations=None, flags=None, along=None, reflectance=None, illumination_weight=0.01,
                           output="numpy", returns="all"):
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
            
        # Change along_centered to be torch.tensor to allow computation
        if output == "torch":
            if torch.is_tensor(self.along_centered) == False:
                self.along_centered = torch.from_numpy(self.along_centered)
                
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

        # Correct for wavefront
        if output == "torch":
            range_corrected = smooth_zi - wavefront_height
        else:
            range_corrected = smooth_zi - wavefront_height
        
        # Determine range
        ranges = self.altimetry.data["heights"][self.location]

        N = ranges.shape[0]
        synth_ranges = ranges
        
        if output == "torch":
            envelope = torch.zeros(N)
            synth_rangepower = torch.zeros((np.unique(self.segment_id).shape[0], N))
        elif output == "numpy":
            envelope = np.zeros(N)
            synth_rangepower = np.zeros((np.unique(self.segment_id).shape[0], N))
        elif output == "jax":
            envelope = jnp.zeros(N)
            synth_rangepower = jnp.zeros((np.unique(self.segment_id).shape[0], N))
            
        f_idx = 0
        for f in np.unique(self.segment_id):
            # Extract the ranges corresponding to the current flag
            range_corrected_z = range_corrected[self.segment_id==f]
            along_centered_z = self.along_centered[self.segment_id==f]
            
            for i in range(N-1):
                if output == "jax":
                    count_z = jnp.logical_and(range_corrected_z < synth_ranges[i],
                                              range_corrected_z > synth_ranges[i+1])
                else:
                    count_z = np.logical_and(range_corrected_z < synth_ranges[i],
                                             range_corrected_z > synth_ranges[i+1])
                    
                if output == "torch":
                    along_dists_z = along_centered_z[np.logical_and(range_corrected_z < synth_ranges[i],
                                                                    range_corrected_z > synth_ranges[i+1]).bool()]
                elif output == "numpy":
                    along_dists_z = along_centered_z[np.logical_and(range_corrected_z < synth_ranges[i],
                                                                    range_corrected_z > synth_ranges[i+1])]
                elif output == "jax":
                    #along_dists_z = jnp.mean(along_centered_z[jnp.logical_and(range_corrected_z < synth_ranges[i],
                    #                                                 range_corrected_z > synth_ranges[i+1])])
                    aoi = jnp.logical_and(range_corrected_z < synth_ranges[i],
                                          range_corrected_z > synth_ranges[i+1])
                    along_dists_z = jnp.where(aoi, along_centered_z, 0)
                        
                # Scale with distance from center
                if output == "torch":
                    scale_param = illumination_weight#0.03 # lower number means more weight to tails
                else:
                    scale_param = illumination_weight#0.03 # lower number means more weight to tails
                if output == "torch":
                    if torch.any(along_dists_z):
                        synth_rangepower[f_idx,i] = count_z.sum()* \
                                                    self.reflectance[f_idx]* \
                                                    illumination(along_dists_z,
                                                                scale_param, 
                                                                mode="normal",
                                                                output="torch")#output)
                    else:
                        synth_rangepower[f_idx,i] = 0
                elif output == "numpy":
                    if np.any(along_dists_z):
                        synth_rangepower[f_idx,i] = np.sum(count_z)* \
                                                    self.reflectance[f_idx].astype(float)* \
                                                    illumination(along_dists_z,
                                                                scale_param, 
                                                                mode="normal",
                                                                output="numpy")#output)
                    else:
                        synth_rangepower[f_idx,i] = 0
                elif output == "jax":
                    synth_rangepower = synth_rangepower.at[f_idx,i].set(jnp.sum(count_z)* \
                                                                        self.reflectance[f_idx].astype(float)* \
                                                                        illumination(along_dists_z,
                                                                                    scale_param, 
                                                                                    mode="normal",
                                                                                    output="jax"))#output)
                        
            # Add individual contribution to the full waveform
            envelope += synth_rangepower[f_idx,:]
            
            f_idx += 1
        # Normalize the individual contributions
        if output == "jax":
            f_idx = 0
            for f in np.unique(self.segment_id):
                synth_rangepower = synth_rangepower.at[f_idx,:].set(synth_rangepower[f_idx,:]/jnp.max(envelope))
                f_idx += 1
        else:
            f_idx = 0
            for f in np.unique(self.segment_id):
                synth_rangepower[f_idx,:] = synth_rangepower[f_idx,:]/envelope.max()
                f_idx += 1
            
        # Smooth the envelope
        n_smooth = 5
        if output == "torch":
            kernel = torch.ones(n_smooth)
            kernel = kernel[None,None,:]
            # Apply smoothing
            envelope = (torch.conv1d(envelope[None,None,:], kernel)/n_smooth)[0,0,:]
        elif output == "numpy":
            envelope = np.convolve(np.ones(n_smooth), envelope, mode="same")/n_smooth
        elif output == "jax":
            envelope = np.convolve(np.ones(n_smooth), envelope, mode="same")/n_smooth
            
        # Normalize the envelope
        envelope = envelope/envelope.max()
        
        match returns:
            case "all":
                match output:
                    case "numpy":
                        return envelope[:256], synth_rangepower[:256], ranges[:256]
                    case "torch":
                        return envelope[:256], synth_rangepower[:256], ranges[:256]
            case "power":
                return envelope[:256]
            case "subpower":
                return synth_rangepower[:256]
            case "ranges":
                return ranges[:256]
            
            
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

    def bayesian_model(self, elevations, flags, segments, obs=None):
        # Bias terms
        N_flags = flags.unique().shape[0]
        N_segments = segments.unique().shape[0]
        
        #### Setup priors
        # Mean vals
        land_mean = 0.0
        water_mean = 0.0
        means = torch.tensor([land_mean, water_mean]).float()
        # SD vals
        land_sd = 0.5
        water_sd = 1.5
        sds = torch.tensor([land_sd, water_sd]).float()
        # Reflection vals
        land_ref = [2,0.1]#[0.5, 1.5]#[1, 2]
        water_ref = [2,0.3]#[1.0, 2.0]#[2, 6]
        refs = torch.tensor([land_ref, water_ref]).float()
            
        # Set values according to majority surface type in segment
        bias_prior = torch.zeros((N_segments, 2))
        ref_prior = torch.zeros((N_segments, 2))
        for s in segments.unique():
            # Get majority surface type
            surface_type = torch.round(torch.mean( flags[segments==s])).int()
            bias_prior[s,:] = torch.tensor([means[surface_type], sds[surface_type]]).float()
            ref_prior[s,:] = torch.tensor([refs[surface_type,0], refs[surface_type,1]]).float()
            
        with pyro.plate("segPlate", N_segments):
            # Bias terms
            bias = pyro.sample("bias", pyro.distributions.Normal(bias_prior[:,0], bias_prior[:,1]))
            
            # Reflectance Terms
            #ref = pyro.sample("ref", pyro.distributions.Uniform(ref_prior[:,0], ref_prior[:,1]))
            ref = pyro.sample("ref", pyro.distributions.Gamma(ref_prior[:,0], ref_prior[:,1]))
        
        # Illumination weight
        #illu = pyro.sample("illu", pyro.distributions.Uniform(0.0001, 0.2))
            
        # Create elevation specific changes
        elev_ = torch.zeros(elevations.shape[0])
        idx = 0
        for f in np.unique(segments):
            elev_[segments==int(f)] += elevations[segments==int(f)] + bias[idx]
            idx += 1
        #elev = pyro.deterministic("elev", elev_) 
        """
        syn_power, syn_subpower, syn_ranges = self.synthetic_waveform(elevations=elev,
                                                                    reflectance=ref,
                                                                    illumination_weight=0.01,
                                                                    output="torch")
        """
        
        syn_power = pyro.sample("syn_power", pyro.distributions.Normal(self.synthetic_waveform(elevations=elev,
                                                                        reflectance=ref,
                                                                        illumination_weight=0.01,
                                                                        output="torch",
                                                                        returns="power"),
                                                                        0.1))
        
        # Get deterministics   
        #syn_subpower = pyro.deterministic("syn_subpower", self.synthetic_waveform(elevations=elev,
        #                                                            reflectance=ref,
        #                                                            illumination_weight=0.01,
        #                                                            output="torch",
        #                                                            returns="subpower"))        
        
        with pyro.plate("observations", syn_power.shape[0]) as x:
            power = pyro.sample("power", pyro.distributions.Normal(syn_power[x], 0.1), obs=obs[x])
        
        #out = {"syn_subpower": syn_subpower,
        #       "syn_elev": elev,
        #       "syn_ranges": syn_ranges}
        return elev#syn_subpower
    
    def bayesian_model_numpyro(self, elevations, flags, segments, obs=None):
        # Bias terms
        N_flags = np.unique(flags).shape[0]
        N_segments = np.unique(segments).shape[0]
        
        #### Setup priors
        # Mean vals
        land_mean = 0.0
        water_mean = 0.0
        means = jnp.array([land_mean, water_mean]).astype(float)
        # SD vals
        land_sd = 0.5
        water_sd = 1.5
        sds = jnp.array([land_sd, water_sd]).astype(float)
        # Reflection vals
        land_ref = [2,0.1]#[0.5, 1.5]#[1, 2]
        water_ref = [2,0.3]#[1.0, 2.0]#[2, 6]
        refs = jnp.array([land_ref, water_ref]).astype(float)
            
        # Set values according to majority surface type in segment
        bias_prior = jnp.zeros((N_segments, 2))
        ref_prior = jnp.zeros((N_segments, 2))
        for s in range(N_segments):
            # Get majority surface type
            surface_type = jnp.round(jnp.mean( flags[segments==s])).astype(int)
            bias_prior = bias_prior.at[s,:].set(jnp.array([means[surface_type], sds[surface_type]]).astype(float))
            ref_prior = ref_prior.at[s,:].set(jnp.array([refs[surface_type,0], refs[surface_type,1]]).astype(float))
            
        with numpyro.plate("segPlate", N_segments):
            # Bias terms
            bias = numpyro.sample("bias", numpyro.distributions.Normal(bias_prior[:,0], bias_prior[:,1]))
            
            # Reflectance Terms
            #ref = pyro.sample("ref", pyro.distributions.Uniform(ref_prior[:,0], ref_prior[:,1]))
            ref = numpyro.sample("ref", numpyro.distributions.Gamma(ref_prior[:,0], ref_prior[:,1]))
        
        # Illumination weight
        #illu = pyro.sample("illu", pyro.distributions.Uniform(0.0001, 0.2))
            
        # Create elevation specific changes
        elev_ = jnp.zeros(elevations.shape[0])
        idx = 0
        for f in range(N_segments):
            elev_ = elev_.at[segments==int(f)].set(elevations[segments==int(f)] + bias[idx])
            idx += 1
        elev = numpyro.deterministic("elev", elev_) 
        """
        syn_power, syn_subpower, syn_ranges = self.synthetic_waveform(elevations=elev,
                                                                    reflectance=ref,
                                                                    illumination_weight=0.01,
                                                                    output="torch")
        """
        
        syn_power = numpyro.sample("syn_power", numpyro.distributions.Normal(self.synthetic_waveform(elevations=elev,
                                                                        reflectance=ref,
                                                                        illumination_weight=0.01,
                                                                        output="jax",
                                                                        returns="power"),
                                                                        0.1))
        #with numpyro.plate("powerPlate", 256):
        #    syn_power = numpyro.sample("syn_power", numpyro.distributions.Normal(0, 2.0))
        # Get deterministics   
        #syn_subpower = pyro.deterministic("syn_subpower", self.synthetic_waveform(elevations=elev,
        #                                                            reflectance=ref,
        #                                                            illumination_weight=0.01,
        #                                                            output="torch",
        #                                                            returns="subpower"))        
        
        with numpyro.plate("observations", syn_power.shape[0]):
            power = numpyro.sample("power", numpyro.distributions.Normal(syn_power, 0.1), obs=obs)
        
        #out = {"syn_subpower": syn_subpower,
        #       "syn_elev": elev,
        #       "syn_ranges": syn_ranges}
        return elev#syn_subpower
    
    def run_inference(self,
                      n_samples=100,
                      n_warmup=50,
                      n_chains=1,
                      disable_progbar=True,
                      save_samples=False,
                      save_predictions=True,
                      savefile_dir = "",
                      engine="pyro"):
        
        print("=========================================")
        print("Prepare data to pyro model")
        # Prepare data for pyro model
        
        # Get waveform at location and shorten to first half
        # to allow quicker computation
        LOCATION = self.location
        heights = self.altimetry.data["heights"][:256][:]
        waveform = self.altimetry.data["power_waveform"][LOCATION]
        waveform = waveform[:256]
        
        match engine:
            case "pyro":
                print("Run Pyro model")
                
                # Convert to torch tensors
                power_train = torch.tensor(waveform/waveform.max()).float()
                elevations_train = torch.tensor(self.elevations).float()
                along_train = torch.tensor(self.along_centered).float()
                flags_train = torch.tensor(self.surface_flags)
                segments_train = torch.tensor(self.segment_id)
                
                N_SAMPLES = n_samples
                N_WARMUP = n_warmup
                N_CHAINS = n_chains
                DISABLE_PROGBAR = disable_progbar
                print("N Samples: {}, N Warmup: {}, N Chains: {}".format(N_SAMPLES,
                                                                        N_WARMUP,
                                                                        N_CHAINS))
                nuts_kernel = NUTS(self.bayesian_model, jit_compile=True, ignore_jit_warnings=True)
                mcmc = MCMC(nuts_kernel, num_samples=N_SAMPLES, warmup_steps=N_WARMUP, 
                            num_chains=N_CHAINS, disable_progbar=DISABLE_PROGBAR)
                mcmc.run(elevations=elevations_train,
                        flags=flags_train,
                        segments=segments_train,
                        obs=power_train)
                
                print(mcmc.summary())
                
            case "numpyro":
                print("Run NumPyro model")
                
                # Get data for training
                power_train = waveform/waveform.max().astype(float)
                elevations_train = self.elevations.astype(float)
                along_train = self.along_centered.astype(float)
                flags_train = self.surface_flags
                segments_train = self.segment_id
                
                # Setup numpyro random key
                rng_key = random.PRNGKey(0)
                rng_key, rng_key_ = random.split(rng_key)
                
                N_SAMPLES = n_samples
                N_WARMUP = n_warmup
                N_CHAINS = n_chains
                DISABLE_PROGBAR = disable_progbar
                print("N Samples: {}, N Warmup: {}, N Chains: {}".format(N_SAMPLES,
                                                                         N_WARMUP,
                                                                         N_CHAINS))
                nuts_kernel = numpyro.infer.NUTS(self.bayesian_model_numpyro)
                mcmc = numpyro.infer.MCMC(nuts_kernel,
                                          num_samples=N_SAMPLES,
                                          num_warmup=N_WARMUP, 
                                          num_chains=N_CHAINS)
                mcmc.run(rng_key_,
                         elevations=elevations_train,
                         flags=flags_train,
                         segments=segments_train,
                         obs=power_train)

                print(mcmc.print_summary())

        # Save samples
        if save_samples == True:
            timestr = datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')
            saved_filename = "{}/mcmc_samples_{}_{}.pt".format(savefile_dir,
                                                                timestr,
                                                                LOCATION)
            torch.save(mcmc.get_samples(), saved_filename)
            print("Saved samples as {}".format(saved_filename))

        # Get predictive statement
        print("Get predictions")
        match engine:
            case "pyro":
                predictive = Predictive(self.bayesian_model, posterior_samples=mcmc.get_samples(), num_samples=N_SAMPLES,
                                        return_sites=("power", "_RETURN"))
                predictive_samples = predictive(elevations=elevations_train,
                                                flags=flags_train,
                                                segments=segments_train,
                                                obs=power_train)
                samples = mcmc.get_samples()
                predictions = {"waveform": predictive_samples["power"].detach().numpy().tolist(),
                            "waveform_syn": samples["syn_power"].detach().numpy().tolist(),
                            "elevations": samples["_RETURN"].detach().numpy().tolist(),
                            "along": along_train.detach().numpy().tolist()
                }       
            case "numpyro":
                predictive = numpyro.infer.Predictive(self.bayesian_model_numpyro, posterior_samples=mcmc.get_samples(), num_samples=N_SAMPLES,
                                                        return_sites=("elev","power", "_RETURN"))
                predictive_samples = predictive(rng_key_,
                                                elevations=elevations_train,
                                                flags=flags_train,
                                                segments=segments_train,
                                                obs=power_train)
                samples = mcmc.get_samples()
                predictions = {"waveform": predictive_samples["power"].tolist(),
                            "waveform_syn": samples["syn_power"].tolist(),
                            "elevations": samples["elev"].tolist(),
                            "along": along_train.tolist()
                }       
        
        # Save predictive statements
        if save_predictions == True:
            current_filename = self.altimetry.filename
            timestr = datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')
            saved_filename = "{}/mcmc_{}_{}_{}.txt".format(savefile_dir,
                                                          current_filename,
                                                          timestr,
                                                          LOCATION)
            #torch.save(predictions, saved_filename)
            with open(saved_filename, "w") as fp:
                json.dump(predictions, fp)
            print("Saved predictive samples as {}".format(saved_filename))
        
        
        
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
    if torch.is_tensor(x) == True:
        return -(torch.sqrt( H**2 - x**2 ) - H)
    else:
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
        elif output == "jax":
            return jnp.exp(-scale_param*(jnp.mean(x)/(1e3))**2)**2
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