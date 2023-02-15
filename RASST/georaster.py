import numpy as np
import matplotlib.pyplot as plt

import rioxarray as rxr
import xarray as xr
from rasterio.crs import CRS

class georaster():
    
    # Setup variables
    dim = []
    data = []
    data_type = "none"
    x = []
    y = []
    extent_geo = []
    extent_idx = []
    bands = {"blue": 1,
             "green": 2,
             "red": 3}
    crs = 'EPSG:4326'
    
    # Setup settings
    print_level = "none" # none/major/info/all
    
    def __init__(self, filename, **kwargs):
        
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
                case "bands":
                    self.bands = kwargs["bands"]
                case "epsg":
                    self.crs = kwargs["epsg"]
                
        # Load raster file and parameters
        self.data = rxr.open_rasterio(filename, masked=True).squeeze() 
        self.printinfo("File from {} was loaded".format(filename))
        
        # Reproject to EPSG:4326 (WGS84)
        self.reproject()
        
        # Initialize the parameters
        self.update_parameters()
        
        # Filetype dependent functionality
        match self.data_type:
            # =======================================================================
            case "optical": # OPTICAL DATA      
                self.printinfo("Case: optical run")
                
                # Convert datavalues
                self.data.values = self.data.values.astype(float)
                
                # Move bands dimension to last element in the array
                self.data = np.moveaxis(self.data.values[:], 0,2)
                self.dim = self.data.shape # update dimensons
            # =======================================================================
            case "dem": # DEM DATA   
                self.printinfo("Case: dem run")
    
    def update_parameters(self):
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
        self.printminor("Updated parameters")
                
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
        self.update_parameters()
        self.printinfo("Georasters combined")
        return self
    
    def create_mask(self, filter="NDWI"):
        """_summary_

        Args:
            filter (str, optional): _description_. Defaults to "NDWI".
        """
        self.data = normalize_image(self.data)
        match filter:
            case "NDWI":
                self.data = NDWI(self.data, self.bands)
                
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
            
    def fastplot(self, bands=[3,2,1], gridsize=100):
        """Make a fast and rough plot of the raster
        """
        raster_plot = self.data
        raster_plot = normalize_image(raster_plot)
        raster_plot = downsampling(raster_plot, gridsize)
        fig, ax = plt.subplots()
        if raster_plot.ndim > 2:
            ax.imshow(raster_plot[:,:,bands], extent=self.extent_idx)
        else:
            ax.imshow(raster_plot[:,:], extent=self.extent_idx)
        plt.show()
        self.printinfo("Plotted downsampled raster")
        
        
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