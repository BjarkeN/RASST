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
    
    # Constants
    SPEED_OF_LIGHT = 299792458 # m/s
    
    def __init__(self, filename=None, l2_filename=None, sat="sen6", **kwargs):
        
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
                        
                    # Convert rangebins to heights
                    self.readgeo(l2_filename)
                        
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
        assert lat_low < lat_high, "ERROR: Resulting extraction is empty!"
        
        alt_output = deepcopy(self)
        
        alt_output.data = sort_dict(alt_output.data, "latitude", 
                                     alt_output.data["latitude"]>lat_low)
        alt_output.data = sort_dict(alt_output.data, "latitude", 
                                     alt_output.data["latitude"]<lat_high)
        
        alt_output.geo = sort_dict(alt_output.geo, "latitude", 
                                     alt_output.geo["latitude"]>lat_low)
        alt_output.geo = sort_dict(alt_output.geo, "latitude", 
                                     alt_output.geo["latitude"]<lat_high)
        
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
    
    def readgeo(self, l2_filename):
        """
        """
        
        dat = nc.Dataset(l2_filename)
        l2_20_keys = ["altitude", "latitude", "longitude","epoch_ocean", "noise_floor_ocean", "range_ocean",
                    "range_ocog","latitude", "ocean_geo_corrections", "index_1hz_measurement", "l1b_record_counter", "l2_record_counter", "ssha"]
        l2_01_keys = ["dac", "distance_to_coast", "geoid", "internal_tide", "inv_bar_cor", "iono_cor_alt_filtered",
                    "l2_record_counter", "load_tide_sol1","load_tide_sol2","mean_dynamic_topography",
                    "model_dry_tropo_cor_measurement_altitude", "model_wet_tropo_cor_measurement_altitude",
                    "pole_tide", "solid_earth_tide", "surface_classification_flag","mean_sea_surface_sol1","mean_sea_surface_sol2"]
        l2_01_keys_ku = ["atm_cor_sig0", "iono_cor_gim","sig0_ocean"]

        #print(dat["data_20"]["ku"]["latitude"][46900])
        #dat["data_20"]["ku"].variables.keys()
        geo = {}
        geo_01 = {}
        geo_01["latitude"] = dat["data_01"]["latitude"][:]
        geo_20 = {}
        for k in l2_20_keys:
            geo_20[k] = dat["data_20"]["ku"][k][:]
            geo[k] = geo_20[k]
        for k in l2_01_keys:
            geo_01[k] = dat["data_01"][k][:]
            geo[k] = np.interp(geo_20["latitude"], geo_01["latitude"],geo_01[k])
        for k in l2_01_keys_ku:
            geo_01[k] = dat["data_01"]["ku"][k][:]
            geo[k] = np.interp(geo_20["latitude"], geo_01["latitude"],geo_01[k])

        self.geo = geo
        print("Geophysical data read from L2 file")
    
    def bin2ranges(self, oversampling = 2):
        """ Determine elevation from rangebins in the power_waveform
        """
        
        # Determine sum of geophysical corrections
        geocorr = np.zeros(self.geo["latitude"].shape)
        for k in ["dac","internal_tide","load_tide_sol1","model_dry_tropo_cor_measurement_altitude","model_wet_tropo_cor_measurement_altitude",\
                "pole_tide","solid_earth_tide","iono_cor_gim"]:
            geocorr += self.geo[k]
            
        # Determine rangebins
        # Create matrix of bin indexes
        n_rows,n_cols = self.data["power_waveform"].shape
        bin_array = np.repeat(np.array([np.arange(n_cols)]),n_rows,0) - 256

        # Create sampling
        # This determines how the rangebins are converted to ranges in meters
        sampling = np.array([self.SPEED_OF_LIGHT/self.data["altimeter_clock"]/(oversampling*2)])
        bin_array = bin_array * sampling.T

        # Move array to center range
        center_range = np.array([self.data["tracker_range_calibrated"]]).T
        bin_array = bin_array + center_range

        # Determine height
        altitude = np.array([self.data["altitude"]]).T
        heights = altitude - bin_array

        # Downsample
        heights = heights.T
        heights = heights - geocorr - self.geo["geoid"]# - geo["mean_dynamic_topography"]
        heights = heights.T
        
        # Create variable with heights
        self.data["heights"] = heights
        
        print("Converted rangebins to heights")
        """ =============== OLD METHOD =================
        assert ("altimeter_clock" in self.keys), "Altimeter clock is missing"
        assert ("power_waveform" in self.keys), "Power Waveform is missing"
        assert ("tracker_range_calibrated" in self.keys), "Tracker range is missing"
        assert ("altitude" in self.keys), "Altitude is missing"
        
        # Create matrix of bin indexes
        n_rows,n_cols = self.data["power_waveform"].shape
        bin_array = np.repeat(np.array([np.arange(n_cols)]),n_rows,0) - 256 - 128
        
        # Create sampling
        # This determines how the rangebins are converted to ranges in meters
        #sampling = np.array([self.data["altimeter_clock"]/oversampling/self.SPEED_OF_LIGHT])
        sampling = np.array([self.SPEED_OF_LIGHT/self.data["altimeter_clock"]/(oversampling*2)])
        #sampling = np.array([self.SPEED_OF_LIGHT/320e6/(oversampling*2)])
        bin_array = bin_array * sampling.T

        # Move array to center range
        center_range = np.array([self.data["tracker_range_calibrated"]]).T
        bin_array = bin_array + center_range
        
        # Determine geophysical corrections
        geocorr = 50
        print("WARNING: Geophysical corrections not implemented yet, using {} m as placeholder".format(geocorr))
        
        # Determine height
        altitude = np.array([self.data["altitude"]]).T
        heights = altitude - bin_array + geocorr
        
        # Create variable with heights
        self.data["heights"] = heights
        
        self.printinfo("Converted rangebins to heights in meters")
        return heights
        """
        
        return heights
        
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