import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import netCDF4 as nc
from copy import deepcopy

class altimetry():
    
    # Settings
    longitude_format = "180" # 360 or 180
    print_level = "info"
    
    # Variables
    keys = []
    data = {}
    info = {}
    fill = {}
    units = {}
    
    def __init__(self, filename=None, sat="sen6", **kwargs):
        
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
                case "longitude_format":
                    self.longitude_format = kwargs["longitude_format"]
                case "keys":
                    self.keys = kwargs["keys"]
                    
                
        if filename != None:
            # Load file
            data_ = nc.Dataset(filename)
            
            # Determine the satellite from which the data is frome
            # (this determines the structure of the data in the nc file)
            match sat:
                case "sen6":
                    # Initialize the parameters
                    if self.keys == []: # Load all keys if none provided
                        self.keys = data_["data_20"]["ku"].variables.keys()
                    
                    # Unwrap file to variables
                    for k in self.keys:
                        self.data[k] = np.ma.getdata(data_["data_20"]["ku"][k][:], subok=False)
                        self.info[k] = data_["data_20"]["ku"][k].comment
                        self.fill[k] = data_["data_20"]["ku"][k][:].fill_value
                        try: # See if there is an attached unit
                            self.units[k] = data_["data_20"]["ku"][k].units
                        except AttributeError:
                            self.units[k] = "None"
                        
                    if self.longitude_format == "180":
                        self.data["longitude"][self.data["longitude"]>180] = self.data["longitude"][self.data["longitude"]>180] - 360
                        
            self.printinfo("File {} loaded".format(filename))
        
    def get_keys(filename, sat="sen6"):
        
        data_ = nc.Dataset(filename)
            
        # Determine the satellite from which the data is frome
        # (this determines the structure of the data in the nc file)
        match sat:
            case "sen6":
                # Initialize the parameters
                keys = data_["data_20"]["ku"].variables.keys()
        return keys
        
    def extract_from_latitude(self,lat_low=-90, lat_high=90):
        """Extract data between low and high latitude

        Args:
            lat_low (int, optional): _description_. Defaults to -90.
            lat_high (int, optional): _description_. Defaults to 90.
        """
        alt_output = deepcopy(self)
        
        alt_output.data = sort_dict(alt_output.data, "latitude", 
                                     alt_output.data["latitude"]>lat_low)
        alt_output.data = sort_dict(alt_output.data, "latitude", 
                                     alt_output.data["latitude"]<lat_high)
        
        self.printinfo("Extracted area between {}N and {}N".format(lat_low,lat_high))
        
        return alt_output
    
    def extract(self, key, logic):
        """Extract data 
        Basically a wrapper for the sort_dict() function
        """
        alt_output = deepcopy(self)
        
        alt_output.data = sort_dict(alt_output.data, key, logic)
        
        self.printinfo("Extracted area of interest")
        
        return alt_output
        
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
        """Make a fast and rough plot
        """
        
        
# ==============================================================================
# INHERITED CLASSES


# ==============================================================================
# FUNCTIONS
        
def sort_dict(dict_input,key,expression):
    '''Remove elements from dict according to an expression
    '''
    dicts = dict_input.copy()
    for k in dicts.keys():
        if k != key:
            dicts[k] = dicts[k][expression]
    dicts[key] = dicts[key][expression]
    return dicts