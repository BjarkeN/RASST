import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import rioxarray as rxr
import xarray as xr
from rasterio.crs import CRS
from copy import deepcopy

class georaster():
    
    # Setup variables
    crs = 'EPSG:4326'
    
    # Setup settings
    print_level = "none" # none/major/info/all
    
    def __init__(self, filename=None, **kwargs):
        
        """_summary_

        Args:
            filename (_type_): _description_
            data_type (str, optional): type of file loaded. Defaults to "simple".
                                  Can be [simple, DEM, optical]
        """
        
        # Go over kwargs
        for arg in kwargs:
            match arg:
                case "data_type":
                    self.data_type = kwargs["data_type"]
                case "printlvl":
                    self.print_level = kwargs["printlvl"]
                case "epsg":
                    self.crs = kwargs["epsg"]
                
        if filename != None:
            # Load raster file and parameters
            self.data = rxr.open_rasterio(filename, masked=True).squeeze() 
            self.printinfo("File from {} was loaded".format(filename))
            
            # Reproject to EPSG:4326 (WGS84)
            self.reproject()
            
            # Initialize the parameters
            self.setup_parameters()
                
    def setup_parameters(self):
        """_summary_
        """
        self.x = self.data.x
        self.y = self.data.y
                
        # Define dimensions
        self.dim = self.data.shape
        
        # Define idx extent
        self.extent_idx = np.array([0, self.data.x.size,
                                    0, self.data.y.size])
        
        # Define geographical extent
        self.extent_geo = np.array([self.data.x[0],
                                    self.data.x[-1],
                                    self.data.y[-1],
                                    self.data.y[1]])
            
        self.printminor("Setup parameters")
                
    def reproject(self):
        """_summary_
        """
        epsg_code =  CRS.from_string(self.crs)
        self.data = self.data.rio.reproject(epsg_code, nodata=np.nan)
        self.printinfo("Reprojected to {}".format(epsg_code))
        
    def concatinate(self, other):
        """_summary_

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.data = xr.concat([self.data, other.data], dim="x")
        self.setup_parameters()
        self.printinfo("Georasters combined")
        return self
        
    def printminor(self, s):
        if self.print_level == "all":
            print("    ",s)
            
    def printmajor(self, s):
        if self.print_level == "major":
            print("  ",s)
            
    def printinfo(self, s):
        if self.print_level == "info":
            print("  ",s)
            
    def fastplot(self, gridsize=100, **kwargs):
        """Make a fast and rough plot of the raster
        """
        band_plots = [3,2,1]
        cmap_setting = "normal"
        cbar = "off"
        
        # Go over kwargs
        for arg in kwargs:
            match arg:
                case "bands":
                    band_plots = kwargs["bands"]
                case "cmap":
                    cmap_setting = kwargs["cmap"]
                case "cbar":
                    cbar = kwargs["cbar"]
                    
        raster_plot = self.data
        raster_plot = normalize_image(raster_plot)
        raster_plot = downsampling(raster_plot, gridsize)
            
        if cmap_setting == "categorical":
            cmap_custom = cm.get_cmap('gist_ncar', len(band_plots))
        elif cmap_setting == "normal":
            cmap_custom = "turbo"
        
        plt.figure()
        if raster_plot.ndim > 2:
            plt.imshow(raster_plot[:,:,band_plots], extent=self.extent_idx,
                      cmap = cmap_custom)
        else:
            plt.imshow(raster_plot[:,:], extent=self.extent_idx,
                      cmap = cmap_custom)
        if cbar == "on":
            plt.colorbar()
        plt.show()
        self.printinfo("Plotted downsampled raster")
        
# ==============================================================================
# INHERITED CLASSES

class dem(georaster):
    """Inherited class from georaster, describing the Digital Elevation Models 
        (DEM)

    Args:
        georaster (_type_): _description_
    """
    def __init__(self, filename=None, **kwargs):
        super().__init__(filename, **kwargs)
        if filename != None:
            self.printinfo("Case: dem run")
        
    def setup_parameters(self):
        super().setup_parameters()
        
class image(georaster):
    """Inherited class from georaster, describing the optical image

    Args:
        georaster (_type_): _description_
    """
    
    bands = {"blue": 1,
             "green": 2,
             "red": 3}
    
    def __init__(self, filename=None, **kwargs):
        super().__init__(filename, **kwargs)
        
        # Go over kwargs
        for arg in kwargs:
            match arg:
                case "bands":
                    self.bands = kwargs["bands"]
        
        if filename != None:
            self.printinfo("Case: optical run")
            
            # Convert datavalues
            self.data.values = self.data.values.astype(float)
            
            # Move bands dimension to last element in the array
            self.data = np.moveaxis(self.data.values[:], 0,2)
            self.dim = self.data.shape # update dimensons
        
    def setup_parameters(self):
        super().setup_parameters()
        
        
    def optical_filter(self, filter="NDWI"):
        """_summary_

        Args:
            filter (str, optional): _description_. Defaults to "NDWI".
        """
        filter_data = self.data
        match filter:
            case "NDWI":
                filter_data = normalize_image(NDWI(filter_data, self.bands))
            case "NDVI":
                filter_data = normalize_image(NDVI(filter_data, self.bands))
                
        filter_output = deepcopy(self)
        filter_output.data = filter_data
                
        return filter_output
    
    def threshold(self, thr_lvl):
        """_summary_

        Args:
            filter (str, optional): _description_. Defaults to "NDWI".
        """
        self.data = (self.data > thr_lvl).astype(int)
        self.data = self.data.astype(float)
       
class mask(georaster):
    """Inherited class from georaster, describing the mask

    Args:
        georaster (_type_): _description_
    """
    
    flags = {"land": 0,
             "water": 1,
             "vegetation": 2}
    
    def __init__(self, filename=None, **kwargs):
        super().__init__(filename, **kwargs)
        
        # Go over kwargs
        for arg in kwargs:
            match arg:
                case "flags":
                    self.bands = kwargs["flags"]
        
        if filename != None:
            self.printinfo("Case: optical run")
            
            # Convert datavalues
            self.data.values = self.data.values.astype(float)
            
            # Move bands dimension to last element in the array
            self.data = np.moveaxis(self.data.values[:], 0,2)
            self.dim = self.data.shape # update dimensons
        
    def setup_parameters(self):
        super().setup_parameters()
        
    def create_from_img(*args):
        """_summary_
        """
        #assert len(args) == 0, "No input to generate mask from"
        
        mask_output = deepcopy(args[0])
        mask_output.data = np.zeros(mask_output.data.shape)
        for id,val in enumerate(args):
            mask_output.data[val.data==1] = id+1
                
        return mask_output
        
# ==============================================================================
# FUNCTIONS
        
def downsampling(im_, gridsize):
    """_summary_

    Args:
        im_ (_type_): _description_
        gridsize (_type_): _description_

    Returns:
        _type_: _description_
    """
    im = np.copy(im_)
    downsample_grid = np.array(np.meshgrid(np.linspace(0,im.shape[1]-1, gridsize).astype(int),
                                    np.linspace(0,im.shape[0]-1, gridsize).astype(int)))
    downsample_grid = np.array([downsample_grid[0].reshape(-1,1), downsample_grid[1].reshape(-1,1)])
    if im.ndim > 2:
        im = im[downsample_grid[1], downsample_grid[0],:].reshape(gridsize,gridsize,-1)
    else: # 2D
        im = im[downsample_grid[1], downsample_grid[0]].reshape(gridsize,gridsize)
    return im
        
def normalize_image(im_,cutoff=99.99,hard_threshold=1e8):
    """Function for normalizing an image for plotting

    Args:
        im_org (_type_): _description_
        cutoff (float, optional): _description_. Defaults to 99.99.
        hard_threshold (_type_, optional): _description_. Defaults to 1e8.

    Returns:
        _type_: _description_
    """
    im = np.copy(im_)
    
    if im.ndim >= 3: # Multi band
        im_original_shape = im.shape
        im = im.reshape((im.shape[0]*im.shape[1],im.shape[2]))
        for i in range(im.shape[-1]):
            im[im[:,i]>hard_threshold, i] = np.nan
            im[im[:,i]>np.nanpercentile(im[:,i],cutoff),i] = np.nan
            im[:,i] = (im[:,i] - np.nanmin(im[:,i])) / (np.nanmax(im[:,i]) - np.nanmin(im[:,i]))
        im = im.reshape(im_original_shape)
        
    elif im.ndim == 2: # Single band
        im_original_shape = im.shape
        im = im.reshape(-1,1)
        im[im>hard_threshold] = np.nan
        im[im>np.nanpercentile(im,cutoff)] = np.nan
        im = (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))
        im = im.reshape(im_original_shape)
    return im

def NDWI(im, bands):
    """Function for calculating the NDWI index
    as defined by McFeeters (1996)
    https://en.wikipedia.org/wiki/Normalized_difference_water_index

    Args:
        im (_type_): _description_
        bands (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Check input
    assert "green" in list(bands.keys()), "Green band is missing"
    assert "nir" in list(bands.keys()), "NIR band is missing"
    
    ndwi = ( im[:,:,bands["green"]] - im[:,:,bands["nir"]] ) / (im[:,:,bands["green"]] + im[:,:,bands["nir"]])
    return ndwi

def NDVI(im, bands):
    """Function for calculating the NDVI index
    as defined by McFeeters (1996)
    https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index

    Args:
        im (_type_): _description_
        bands (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Check input
    assert "red" in list(bands.keys()), "Red band is missing"
    assert "nir" in list(bands.keys()), "NIR band is missing"
    
    ndwi = ( im[:,:,bands["nir"]] - im[:,:,bands["red"]] ) / (im[:,:,bands["nir"]] + im[:,:,bands["red"]])
    return ndwi