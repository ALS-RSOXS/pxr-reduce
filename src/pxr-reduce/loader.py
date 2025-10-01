# Basic modules
import warnings
import copy
import re
import pathlib

# Math libraries
import numpy as np
import pandas as pd
from scipy.odr import ODR, Model, RealData # For finding stitch ratio

# Image libraries
from astropy.io import fits # To load .fits files

# Plotting libraries
import matplotlib.pyplot as plt
import matpotlib.colors as mpl_colors

from utils import attributes, image, units

default_cmap = "terrain"

# Dictionary to convert motor names to standard nomenclature
header_names = {
    "fits_index": "fits_index",
    "Beamline Energy": "energy",
    "EPU Polarization": "polarization",
    "Sample Theta": "sam_th",
    "CCD Theta": "det_th",
    "Sample X": "sam_x",
    "Sample Y": "sam_y",
    "Sample Z": "sam_z",
    "image": "raw_image",
    "EXPOSURE": "exposure",
    "Higher Order Suppressor": 'hos',
    "Upstream JJ Vert Aperture": 'slits_vert',
    "Upstream JJ Horz Aperture": 'slits_horz',
    "Beam Current": 'beam_current',
    "AI 3 Izero": 'i0',
}

# Energy is a special case for 0.05 values and will be slightly different
header_resolutions = {
    "exposure": 4, 
    "sam_th": 4, 
    "det_th": 4,
    "sam_x": 4,
    "sam_y": 4,
    "hos": 1,
    "polarization": 0,
    "slits_vert": 4,
    "slits_horz": 4,
}

stitch_motors = ['sam_th']

class PrsoxrLoader:
    """
    Loader for PRSoXR data from beamline 11.0.1.2 taken with the CCD camera.
    
    Parameters
    -----------
    files : list
        List of .fits to be loaded. Include full filepaths
        
        >>> # Recommended Usage
        >>> import pathlib
        >>> path_s = pathlib.Path('../ALS/2020 Nov/MF114A/spol/250eV')
        >>> files = list(path_s.glob('*fits')) # All .fits in path_s
        
        The newly created 'files' is now a list of filepaths to each reflectivity point.

    AI_file: pathlib.Path
        A filepath to a complimentary .txt file that contains updated metadata. 
        This is only used for data collected with the 'Beamline Scan Panel' at BL11012.
        
    auto_load: bool
        Should the images be loaded automatically upon creating object?
        
    **kwargs:
        process_vars that want to be updated at the time of creation.
        
    Attributes
    -----------
    exposure_offset: float [s]
        Offset to add to camera exposure time. Time it takes to physically open and close shutter.
        This should be measured in advanced and not changed often.
    energy_resoltion: float 
        Energy will be normalized based on the following equation: 
            np.round(self.data['energy']*energy_resolution)/energy_resolution
        Enables rounding to non-integer values. Default rounds to 0.05 eV
    sam_th_offset: float [th]
        Offset added to sam_th at the time of measurement default is None
    sam_th_correction: Bool
        Default is True. It will determine the sam_th_offset based on the initial measurement positions
    energy_offset: float [eV]
        Optional offset to the energy value. Defaults at 0
    det_pixel_size: float [mm/pixel]
        Size of detector pixel. May change as detectors change
    roi_height: int
        Size of the ROI used to integrate over beam spot. Vertical dimension
    roi_width: int
        Size of the ROI used to integrate over beam spot. Horizontal Dimension
    trim_x: int
        Number of pixels on the edge of the detector to remove fromm consideration
    trim_y: int
        Number of pixels on the edge of the detector (vertical) to remove from considerations
    stitch_cutoff: float ['ratio']
        Used to identify positions at which a 'stitch' has occured between the data.
    drop_failed_stitch: bool
        Decision on whether to drop all datapoints that did not stitch correctly.
        Defaults to True, a warning will be given
    stitch_mark_tol: float
        Value used to verify whether or not a tracked motor for stitching has moved
    dark_pix_offset: int [pixels]
        Number of pixels to offset the region used for dark subtraction from the edge of the frame
    new_scan_marker: float [deg]
        How far the 'sam_th' motor needs to move in order to indicate a new 'scan' starting from 0
    drift_distance: int [pixels]
        Distance that the beam can drift from its nominal positions
    mask_threshold: int [counts]
        Threshold used to identify where data-points are potentially located
    filter_size: int
        Filter size for the zinged image
    darkside: 'RHS' or 'LHS'
        Side of the image to collect the dark image.
    saturate_threshold: int [counts]
        A value to indicate whether an image has been saturated or not.
        It will check how close the maximum intensity is to 2**16.
    
        
    Notes
    ------
    
    Print the loader to view variables that will be used in reduction. Update them using the attributes listed in this API.
    
    >>> loader = PrsoxrLoader(files, name='MF114A_spol')
    >>> print(loader) #Default values
        Sample Name - MF114A
        Number of scans - 402
        ______________________________
        Reduction Variables
        ______________________________
        Shutter offset = 0.00389278
        Sample Location = 0
        Angle Offset = -0.0
        Energy Offset = 0
        SNR Cutoff = 1.01
        ______________________________
        Image Processing
        ______________________________
        Image X axis = 200
        Image Y axis = 200
        Image Edge Trim = (5, 5)
        Dark Calc Location = LHS
        Dizinger Threshold = 10
        Dizinger Size = 3
    >>>loader.shutter_offset = 0.004 #Update the shutter offset

        
    mask : np.ndarray (Boolean)
        Array with dimensions equal to an image. Elements set to `False` will be excluded when finding beamcenter.
        
        >>> # Recommended usage
        >>> loader = loader = PrsoxrLoader(files)
        >>> mask = np.full_like(loader.images[0], True, dtype=bool)
        >>> mask[:50,:50] = False # block out some region
        >>> loader.mask = mask
    >>>
        
    Once process attributes have been setup by the user, the function can be called to load the data. An ROI will need
    to be specified at the time of processing. Use the ``self.check_spot()`` function to find appropriate dimensions.
    
    >>> refl = loader(h=40, w=30)
    
    Data that has been loaded can be exported using the ``self.save_csv(path)`` and ``self.save_hdf5(path)`` functions.
    
    """

    process_vars = {
        'exposure_offset': 0.00389278, # [s]
        'energy_resolution': 20 # 0.05eV step resolution
        'sam_th_offset': None, # [deg]
        'sam_th_correction': True,
        'energy_offset': 0, # [eV]
        'det_pixel_size': 0.027, #[mm/pixel]
        'roi_height': 10, # [pixels]
        'roi_width': 10, # [pixels]
        'trim_x': 10, # [pixels]
        'trim_y': 10, # [pixels],
        'stitch_cutoff': 1.003, # [ratio]
        'drop_failed_stitch': True,
        'stitch_mark_tol': 1e-5, #[unitless]
        'dark_pix_offset': 20, # [pixels]
        'new_scan_marker': 5, # [mm] # Way to indicate a new sample by z-motion
        'drift_distance': 25, # [pixels] # Distance that the beam can drift from the nominal positions
        'mask_threshold': 800, # [counts] # Counts that indicate an easy spot for masking
        'filter_size': 3,
        'darkside': "LHS",
        'reprocess_vars': True,
        'saturate_threshold': 2
    }

    def __init__(self, files, AI_file = None, auto_load=False, energy_resolution=20, **kwargs):
        # Update the process variables with any initial conditions
        self.process_vars = self.process_vars.copy()
        self.process_vars.update(kwargs)
        
        # Check type of files of input---
        self.files = []
        # Is files empty?
        if len(files) == 0:
            print(f"The 'files' input is empty. Nothing can be loaded.")
            print(f"Check your directory for FITS files.")
            return 0
        # Is files a single list?
        if isinstance(files, (str, pathlib.Path)):
            print(f"A single file is being loaded. This will not process correctly as a RSoXR experiment.")
            path_list = [files]
        elif isinstance(files, list):
            path_list = files
        else:
            msg = "PrsoxrLoader was not given a correct input. Only paths to FITS files are currently accepted."
            raise ValueError(msg)
            return 0
        for fp in path_list:
            file = pathlib.Path(fp)
            if not file.is_file():
                msg = f"{file} is not a valid file."
                raise FileNotFoundError(msg)
            if file.suffix != ".fits":
                msg = f"{file} is not a FITS file."
                raise ValueError(msg)
            self.files.append(file)
            
        # Check AI File
        if isinstance(AI_file, (str, pathlib.Path)):
            print(f"Loading AI-file to supplement FITS meta-data")
            AI_file = pathlib.Path(AI_file)
            if not AI_file.is_file():
                msg = f"{AI_file} is not a valid file."
                raise FileNotFoundError(msg)
            if AI_file.suffix != ".txt":
                msg = f"{AI_file} is not a txt file, it is unlikely the correct AI companion file."
                raise ValueError(msg)
        else:
            AI_file = None

        # Load the files into the Loader
        tmp = []

        # Get information about the sample / path from the first fits file
        path0 = self.files[0]
        self.path = path0.parent
        self.name = re.search(r'^(.*)[ _](\d+)\.fits$', path0.name).group(1)
        
        for i, fits in tqdm(enumerate(self.files), "Loading .fits", total=len(self.files)):
            # Collect information about the filepath to save -- 
            fits_name = fits.name # The name of the current file
            fits_index = int(re.search(r'[ _](\d+)\.fits$', fits_name).group(1)) # Index of the file (if it gets messed up for some reason)
            # Load the data
            df_fits = dict_load_fits(fits) # Load .fits files into a dictionary
            if AI_file is not None: #Only run if the meta-data needs to be reuplodaed from the .txt file
                temp_meta = self.load_AI_meta(file=AI_file)
                df_fits = self.update_meta(df_fits, temp_meta.iloc[fits_index]) # get the correct line item in the AI file --
            df_fits['fits_index'] = fits_index # Save the index
            tmp.append(df_fits) # save the file
        data_dict = {key: [d[key] for d in tmp] for key in tmp[0].keys()}
        df = pd.DataFrame(data_dict)
        
        # Rename the files and only extract those that matter--
        self.data = df[list(header_names.keys())].rename(columns=header_names).round(header_resolutions)
        self.data['energy'] = np.round(self.data['energy']*self.process_vars['energy_resolution'])/self.process_vars['energy_resolution'] # round energy to the nearest 0.25 eV
        
        # Sort the inputted files are not in order of the file index
        self.data = self.data.sort_values('fits_index', ignore_index=True)
        self.data.insert(1, 'scan', 0) # populate a new volumn that identifies the individual scans (TBD)

        # Other useful columns
        self.data['mask'] = None
        self.data['beam_spot'] = None

        # Has the data been processed?
        self.data_processed = False
        if auto_load:
            self.reprocess_images()
            
    def __str__(self):
        s = []
        s.append("PRSoXR Loader:")
        s.append(f"Sample Name: {self.name}")
        s.append(f"Sample Path: {self.path}")
        s.append(f"Number of Scans: {len(self)}")
        return "\n".join(s)

    def __len__(self):
        return len(self.files)

    @property
    def meta(self):
        meta_cols = [header_names[k] for k in header_names if 'image' not in k]
        return self.data[meta_cols]

    def __call__(self):
        return self.calc_refl()
        

    def load_AI_meta(self, file=None):
        # Load .txt file here to supplement metadata if needed
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if "DATA" in line:
                    header_line = i
        return pd.read_csv(file, skiprows=header_line, header=1, delimiter="\t")###################
        
    def update_meta(self, meta_fits, meta_AI):
        # Update the metadata from the AI text file
        values_to_replace = [
            'Sample Theta',
            'Sample X',
            'Sample Y',
            'Sample Z',
            'CCD Theta',
            'Higher Order Suppressor',
            'Beamline Energy'
        ]
        for value in values_to_replace:
            meta_fits[value] = meta_AI[value+' Actual']
        # Other monitors
        meta_fits['Beam Current'] = meta_AI['Beam Current']
        meta_fits['AI 3 Izero'] = meta_AI['AI 3 Izero']
        return meta_fits

    def reprocess_images(self):
        self.cleanup_metadata()
        self.clean_images() # Filter and zing the images for further reduction. Only do this once.
        self.generate_series_mask() # Create a mask based on the total drift of teh beam and a radial threshold.
        #self.locate_beam_byscan()
        self.process_images()
        self.reprocess_vars = False

    def cleanup_metadata(self):
        # Check monitors that may be equal to zero
        self.data['beam_current'] = self.data['beam_current'].apply(lambda x: x if x > 50 else 1) 
        self.data['i0'] = self.data['i0'].apply(lambda x: x if x > 0 else 1) 
        self.data['exposure'] = self.data['exposure'].apply(lambda x: x+self.process_vars['exposure_offset'] if x > 0 else 1) 
                
        # Label each sample in terms of the 'scan'. different scans can have the same energy/pol/steps
        new_scan_markers = np.abs(self.data['sam_th'].diff()) > self.process_vars['new_scan_marker']
        self.data['scan'] = new_scan_markers.cumsum()

        # Add offsets
        # Energy
        self.data['energy'] += self.process_vars['energy_offset']
        # Theta
        if self.process_vars['sam_th_offset'] == None and self.process_vars['sam_th_correction']: # Was a sample theta offset given to the loader?
            #[data['sam_z'][np.abs(data['sam_z'].diff()) > 0].index+1]
            sam_z_move = np.abs(self.data['sam_z'].diff()) > 0 # First move of sam_z to return to blocking the beam
            sam_z_move_index = self.data['sam_z'][sam_z_move].index+1
            begin_refl_angle = self.data['sam_th'][sam_z_move_index] # First angle to collect data
            begin_ccd_angle = self.data['det_th'][sam_z_move_index] # First detector angle to collect data
            sam_th_offset = np.round((begin_ccd_angle/2 - begin_refl_angle).iloc[0],4) # Should be in th-2th configuration
            self.data['sam_th'] += sam_th_offset # Correct for offset
            self.process_vars['sam_th_offset'] # Save offset
            print(f"----")
            print(f"sam_th offset not given.")
            print(f"Assuming a th-2th geometry -> Offset determed to be sam_th_offset = {sam_th_offset} [deg]")
            print(f"Applying offset to data")
            print(f"Reprocess data with 'loader.process_vars['sam_th_correction']=False' to prevent automatic offset")
            print(f"----")
        # Now that energy and theta are calculated correctly, make the q-col
        self.data['wavelength'] = self.data['energy'].apply(lambda x: energy_to_wavelength(x))
        self.data['q'] = self.data.apply(lambda df: theta_to_q(np.deg2rad(df['sam_th']), df['wavelength']), axis=1)

    def clean_images(self):
        
        # Filter the image using a median filter to highlight beam spot
        tqdm.pandas(desc="Filtering images")
        self.data['filtered_image'] = self.data.progress_apply(
            lambda df: median_filter(df['raw_image'], size=self.process_vars['filter_size']),
            axis=1
        )
        # Dezinger the images
        self.data['zinged_image'] = self.data.apply(
            lambda df: dezinger_image(df['raw_image'], med_result=df['filtered_image']),
            axis=1,
        )

    def generate_series_mask(self):
        # Get parameters for reduction
        threshold = self.process_vars['mask_threshold']
        r = self.process_vars['drift_distance'] # The distance that you will allow for the beam to move between each photo
        # Identify all the locations where the beam will likely be located. Hot spots are eay to grab
        mask_temp = self.data['zinged_image'].copy().sum()/len(self.data['zinged_image'])

        mask_loc = np.argwhere(
            (mask_temp > threshold)
        ) 
        grid_map = np.indices(mask_temp.shape) # Get a grid
        mask = np.zeros_like(mask_temp, bool) # Initially mask all points
        for xy in mask_loc:
            distance_mask = np.sqrt((grid_map[0]-xy[0])**2 + (grid_map[1]-xy[1])**2) < r # Go point-by-point and find the local 10 pixel circle
            mask = mask | distance_mask # Add those points to be integrated.

        self.mask = mask # Mask does not move outside of this initial area. No odd jumps.
        
        
    # Calculate filtered and zinged images for later use.
    def process_images(self):
        # Cleanup some potential issues if parameters did not load
        dx = self.process_vars['trim_x'] # How much to trim from the edges of the image (x dimension)
        dy = self.process_vars['trim_y'] # How much to trim from the edges of the image (y dimension)
        process_mask = self.mask[dx:-dx, dy:-dy]
        
        self.data['reduced_image'] = self.data.apply(
            lambda df: df['zinged_image'][dx:-dx, dy:-dy],
            axis=1
        )
        
        self.data['beam_spot'] = self.data.apply(
            lambda df: np.unravel_index(
                np.argmax(np.where(process_mask, df['reduced_image'],0)),
                df['reduced_image'].shape
            ), 
            axis=1
        )

        self.data['roi'] = self.data.apply(
            lambda df: self.update_roi(df['beam_spot']),
            axis=1,
        )
        # Calculate  ROI of the 'dark' region. One to the left and right of the beamspot
        self.data['roi_d'] = self.data.apply(
            lambda df: self.update_dark_roi(df['beam_spot']),
            axis=1
        )
        # Find the locations of the collected image to integrate for beam and the dark frame
        self.data['spot'] = self.data.apply(
            lambda df: df['reduced_image'][df['roi']],
            axis=1
        )

        self.data['dark'] = self.data.apply(
            lambda df: df['reduced_image'][df['roi_d']],
            axis=1
        )
        # Sum up all the pixels within the roi's that amount to a 'point detector'
        self.data['counts_spot'] = self.data.apply(
            lambda df: df['spot'].sum(),
            axis=1
        )
        self.data['counts_dark'] = self.data.apply(
            lambda df: df['dark'].sum(),
            axis=1
        )
        # Subtract and calculate the intensity
        self.data['counts_refl'] = self.data.apply(
            lambda df: (df['counts_spot'] - df['counts_dark']) /
            (
                df['exposure']*df['beam_current']
            ),
            axis=1
        )
        self.data['counts_err'] = self.data.apply(
            lambda df: (df['counts_spot'] - df['counts_dark']) /
            (
                df['exposure']*df['beam_current']
            ),
            axis=1
        )
        # Calculate ratios to determine stitch viability
        self.data['counts_ratio'] = self.data.apply(
            lambda df: (df['counts_spot']/df['counts_dark']),
            axis=1
        )

        self.data['is_saturated'] = self.data.apply(
            lambda df: self.check_saturation(df['reduced_image'], threshold=self.process_vars['saturate_threshold']),
            axis=1
        )

        #self.data.drop(['raw_image'], axis=1) # remove extra data -- 

        # End here
        self.data_processed = True


    def update_roi(self, beam_spot): # Calculate the beam dimensions
        h = self.process_vars['roi_height']
        w = self.process_vars['roi_width']
        # slice the roi
        x_low_bound = beam_spot[1] - w//2
        y_low_bound = beam_spot[0] - h//2
        slx = slice(
            x_low_bound,
            x_low_bound + w
        )
        sly = slice(
            y_low_bound,
            y_low_bound + h
        )
        
        return (sly, slx)
        
    def update_dark_roi(self, beam_spot):
        h = self.process_vars['roi_height']
        w = self.process_vars['roi_width']
        offset = self.process_vars['dark_pix_offset']
        darkside = self.process_vars['darkside']
        # Find which side of the spot to collect as dark
        if beam_spot[1] - 3*w//2 - offset > 0 and darkside=='LHS':
            x_low_bound = beam_spot[1] - w//2 - w - offset
        else:
            x_low_bound = beam_spot[1] + w//2 + offset
            
        y_low_bound = beam_spot[0] - h//2
        slx = slice(
            x_low_bound,
            x_low_bound + w
        )

        sly = slice(
            y_low_bound,
            y_low_bound + h
        )

        return (sly, slx)        
    
    def subtract_background(self, image, dark):
        bkg_sub = self.process_vars['bkg_sub']
        left = image[:, :bkg_sub]
        right = image[:, -bkg_sub:]
        if left.sum() <= right.sum():
            image = image - right.mean(axis=1)[:, None]
        else:
            image = image - left.mean(axis=1)[:, None]
        return image

    def check_saturation(self, image, threshold=1):
        maxcounts = image.max()
        if ((2**16 - maxcounts) < threshold):
            return True
        else:
            return False
            
    def locate_beam_byscan(self, save_mask=True):
        current_scan = 0
        prev_beamspot = None
        
        for i in tqdm(range(len(self)), desc="Finding Beam centers:"):
            scan = self.data['scan'].iloc[i]
            w = self.process_vars['roi_width']
            h = self.process_vars['roi_height']
            if scan > current_scan or i==0:
                current_scan = scan
                beam_spot, mask = self.find_maximum(self.data['filtered_image'].iloc[i], init=None, w=w, h=h)
            else:
                prev_beamspot = self.data['beam_spot'].iloc[i-1]
                beam_spot, mask = self.find_maximum(self.data['filtered_image'].iloc[i], init=prev_beamspot, w=w, h=h)
            if save_mask:
                self.data.at[i, 'beam_spot'] = beam_spot
                self.data.at[i, 'mask'] = mask
            else:
                self.data.at[i, 'beam_spot'] = beam_spot

    def find_maximum(self, init_image, init=None, w=20, h=20): # Calculate the spot
        r = self.process_vars['drift_distance'] # The distance that you will allow for the beam to move between each photo

        if w or h:
            img_height, img_width = init_image.shape[:2]

            # Define the starting full slice for each dimension
            y_start, y_end = 0, img_height
            x_start, x_end = 0, img_width

            if h:
                offset_y = (h // 2) + 1
                y_start, y_end = offset_y, img_height - offset_y
        
            if w:
                offset_x = (w // 2) + 1
                x_start, x_end = offset_x, img_width - offset_x

            # Apply the final slice
            image = init_image[x_start:x_end, y_start:y_end]
        else:
            image = init_image
        
        if isinstance(init, tuple):
            grid_map = np.indices(image.shape)
            distance_mask = np.sqrt((grid_map[0]-init[0])**2 + (grid_map[1]-init[1])**2) < r
            masked_image = np.where(distance_mask, image, 0)
            yloc, xloc = np.unravel_index(np.argmax(masked_image), masked_image.shape) # Get the peak position of the direct beam, easy  

        else:
            distance_mask = np.ones_like(image, dtype=bool)
            yloc, xloc = np.unravel_index(np.argmax(image), image.shape) # Get the peak position of the direct beam, easy

        # Adjust for the actual zinged image to integrate.
        xloc += w//2
        yloc += h//2
        return (yloc, xloc), distance_mask


    def check_spot(self, fits_index, d=1):
        if fits_index not in self.data['fits_index']:
            print(f"FITS file not found in loaded data, please verify index.")
            return 0
        elif not self.data_processed:
            print(f"Data has not yet been processed. Please run 'loader.reprocess_images()'.")
            return 0
        elif self.process_vars["reprocess_vars"]:
            self.cleanup_metadata()
            #self.generate_series_mask()
            self.process_images()
            self.reprocess_vars = False

        df = self.data[self.data['fits_index'] == fits_index]
        
        # Relevant outputs that you would want to read.
        print('Relevant Motor Positions --------')
        print('Angle Theta: {} [th]'.format(df['sam_th'].iloc[0]))
        print('CCD Theta: {} [th]'.format(df['det_th'].iloc[0]))
        print('Photon Energy: {} [eV]'.format(df['energy'].iloc[0]))
        print('Polarization: {}'.format(df['polarization'].iloc[0]))
        print('Higher Order Suppressor: {} [th]'.format(df['hos'].iloc[0]))
        print('Exposure: {} [s]'.format(df['exposure'].iloc[0]))
        print('Beam Current: {} [mA]'.format(df['beam_current'].iloc[0]))
        print('AI 3 Izero: {} [nA]'.format(df['i0'].iloc[0]))
        print('Upstream JJ Vert Aperture: {} [mm]'.format(df['slits_vert'].iloc[0]))
        print('Upstream JJ Horz Aperture: {} [mm]'.format(df['slits_horz'].iloc[0]))
        print('\n')
        print('Processed Variables:')
        print('Q: {} [A-1]'.format(df['q'].iloc[0]))
        print('Specular: {} [counts]'.format(df['counts_spot'].iloc[0]))
        print('Background: {} [counts]'.format(df['counts_dark'].iloc[0]))
        print('Signal: {}'.format(df['counts_refl'].iloc[0]))
        print('SNR: {}'.format(df['counts_ratio'].iloc[0]))
        print('Beam center (x,y): {} [pixels]'.format( df['beam_spot'].iloc[0]))
        print('Highest Pixel Counts: {} [counts]'.format(df['zinged_image'].iloc[0].max()))

        if d:

            fig, ax = plt.subplots(3, 2, subplot_kw={'xticks': [], 'yticks': []})#, figsize=(12, 8))
            ax[0,0].imshow(df['zinged_image'].iloc[0], norm=mpl_colors.LogNorm(), cmap=default_cmap)
            ax[0,0].set_title('Raw Image')
            ax[0,1].imshow(df['filtered_image'].iloc[0], norm=mpl_colors.LogNorm(), cmap=default_cmap)
            ax[0,1].set_title('Filtered Image')

            # Mask if necessary
            if self.mask is not None:
                mask_display = np.ma.masked_where(self.mask == True, self.mask)
                ax[0,0].imshow(mask_display, cmap='Greys_r')
                ax[0,1].imshow(mask_display, cmap='Greys_r')

            # Add rectangles on each plot
            #ax[0,0].add_patch(rect_spot)
            #ax[0,0].add_patch(rect_dark)
            ax[0,1].plot(df['beam_spot'].iloc[0][1], df['beam_spot'].iloc[0][0], 'ro', ms=2)

            ax[1,0].imshow(df['spot'].iloc[0], cmap=default_cmap)
            ax[1,0].set_title('Beam Shape lin-scale')

            ax[1,1].imshow(df['dark'].iloc[0], cmap=default_cmap)
            ax[1,1].set_title('Dark Image lin-scale')
            
            ax[2,0].imshow(df['spot'].iloc[0], norm=mpl_colors.LogNorm(), cmap=default_cmap)
            ax[2,0].set_title('Beam Shape log-scale')

            ax[2,1].imshow(df['dark'].iloc[0], norm=mpl_colors.LogNorm(), cmap=default_cmap)
            ax[2,1].set_title('Dark Image log-scale')

            plt.tight_layout()
            plt.show()
    def display_fits(self, fits_index, bs=True):
        
        df = self.data[self.data['fits_index'] == fits_index]

        fig, ax = plt.subplots(3, 2, subplot_kw={'xticks': [], 'yticks': []})#, figsize=(12, 8))
        ax[0,0].imshow(df['zinged_image'].iloc[0], norm=mpl_colors.LogNorm(), cmap=default_cmap)
        ax[0,0].set_title('Raw Image')
        ax[0,1].imshow(df['filtered_image'].iloc[0], norm=mpl_colors.LogNorm(), cmap=default_cmap)
        ax[0,1].set_title('Filtered Image')

        # Mask if necessary
        if self.mask is not None:
            mask_display = np.ma.masked_where(self.mask == True, self.mask)
            ax[0,0].imshow(mask_display, cmap='Greys_r')
            ax[0,1].imshow(mask_display, cmap='Greys_r')

        # Add rectangles on each plot
        #ax[0,0].add_patch(rect_spot)
        #ax[0,0].add_patch(rect_dark)
        if bs:
            ax[0,1].plot(df['beam_spot'].iloc[0][1], df['beam_spot'].iloc[0][0], 'ro', ms=2)
    
            ax[1,0].imshow(df['spot'].iloc[0], cmap=default_cmap)
            ax[1,0].set_title('Beam Shape lin-scale')
    
            ax[1,1].imshow(df['dark'].iloc[0], cmap=default_cmap)
            ax[1,1].set_title('Dark Image lin-scale')
            
            ax[2,0].imshow(df['spot'].iloc[0], norm=mpl_colors.LogNorm(), cmap=default_cmap)
            ax[2,0].set_title('Beam Shape log-scale')
    
            ax[2,1].imshow(df['dark'].iloc[0], norm=mpl_colors.LogNorm(), cmap=default_cmap)
            ax[2,1].set_title('Dark Image log-scale')

        plt.tight_layout()
        plt.show()


    def calc_refl(self, drop_duplicates=True):
        """
        Function that performs a data reduction of PRSOXR data.
        """

        # Verify that the data has been processed and the metadata does not need to be recalculated
        if not self.data_processed:
            print(f"Data has not yet been processed. Please run 'loader.reprocess_images()'.")
            return 0
        elif self.process_vars["reprocess_vars"]:
            self.cleanup_metadata()
            #self.locate_beam_byscan()
            self.process_images()
            self.reprocess_vars = False

        if self.data['is_saturated'].sum() > 0:
            warnings.warn(f"The CCD was likely saturated during data collection. Stitching may be impacted")
            
        scans = self.data['scan'].iloc[-1] + 1 # How many total scans are included in each calculation? (starts at 0)
        self.data = self.data.groupby('scan').apply(self.normalize_scan, include_groups=False).reset_index(level='scan')
        self.data = self.data.groupby('scan').apply(self.find_stitch_points, include_groups=False).reset_index(level='scan')
        self.data = self.data.groupby('scan').apply(self.calc_scale_factors, include_groups=False).reset_index(level='scan')
        self.data['R'] = self.data.apply(lambda df: df['R']/df['scale'], axis=1)
        #self.data['R_err'] = self.data.apply(
        #    lambda df: df['R'] * ((df['R_err']/df['R'])**2 + (df['scale_err']/df['scale'])**2)**0.5,
        #   axis=1
        #)
        # Generate output mask
        mask = (self.data['i0_mask']<1)
        mask &= (self.data['is_saturated']==False)
        mask &= (self.data['R'] > 0)
        if self.process_vars['drop_failed_stitch']:
            mask &= (self.data['failed_stitch_mask'] < 1)
            
        out = self.data[mask][['scan', 'energy', 'polarization', 'sam_th', 'q', 'R', 'R_err']]
        if drop_duplicates:
            out = out.groupby(["sam_th", 'energy', 'polarization'], as_index=False).mean()
        
        return out

    def normalize_scan(self, df):
        # Normalize data to the i0 given by the 'sam_z' position
        try:
            i0_cutoff = df.loc[np.abs(df['sam_z'].diff()) > 0.1].index[0]+1 # The first position where we move Z into the beam
            i0 = df['counts_refl'].loc[:i0_cutoff].mean()
            i0_err = df['counts_err'].loc[:i0_cutoff].std()
        except IndexError:
            print(f"No direct beam found for scan#{df['fits_index'].iloc[0]}. Not normalizing the beam.")
            i0 = 1
            i0_err = 0
            i0_cutoff = 0
        # Incorporate stability into the beam

        df['R'] = df['counts_refl']/i0
        df['R_err'] = np.sqrt((df['counts_refl'] / i0) ** 2 * ((df['counts_err'] / df['counts_refl']) ** 2 + (i0_err / i0) ** 2))
        df['i0_mask'] = (df.index < i0_cutoff).astype(int)
        return df

    def find_stitch_points(self, df):
        igroup = df.index[0] # fixes issue with .groupby functions
        df['mark'] = None # Add index to watch for changes
        df['motor'] = None # Add something to record the motor that changes (only 1)

        # Index counting
        skip = False
        skip_count = 0
        skip_count_reset = 2
        # Runt through each motor that chnges
        for motor in stitch_motors:
            if 'sam_th' in motor: # This is the only motor that it matters which direction it moves
                iterfunc = np.diff(df[motor]) < -0.2
            else:
                iterfunc = np.diff(df[motor])
            for i, val in enumerate(iterfunc):
                if skip:
                    if skip_count <= skip_count_reset:
                        skip_count += 1
                    else:
                        skip=False
                        skip_count = 0
                elif abs(val) > self.process_vars['stitch_mark_tol']:
                    if df['mark'].iloc[i] == None:
                        df.loc[igroup+i+1, 'mark'] = 1 # adds to the 'next' position
                        df.loc[igroup+i+1, 'motor'] = motor
                    skip=True
                    
        return df 

    def calc_scale_factors(self,df):
        igroup = df.index[0] # fixes issue with .groupby functions
        # Initialze the new columns here
        df['scale'] = 1.0
        df['scale_err'] = 0.0
        df['num_stitch_points'] = 0
        df['safe_stitch_values'] = None
        df['failed_stitch_mask'] = 0 

        # Get the model for the stitch ratio
        linear = Model(stitchratio)
        prev_mark_index = 0
        for i in range(len(df)):# row in df.iterrows():
            if df['mark'].iloc[i] == None:
                continue

            # If here, we are at a stitch point
            sam_th_stitch = df['sam_th'].iloc[i]
            # Calculate the number of datapoints are repeated.
            repeat = 0
            for j, val in enumerate(df['sam_th'].iloc[i:]):
                if val == sam_th_stitch: # I have entered a region where qstitch matches the value
                    repeat += 1
                else: # I am no longer at the same repeated value
                    break
            # Create a few dummy lists for each stitch point
            ipre = []
            ipost = []
            # Cycle back and find the indices where stitch points will be calculated.
            # Only start at the prev mark point to avoid edge cases
            for j, val in enumerate(df['sam_th'].iloc[prev_mark_index:i]): # Start cycling from the beginning
                jj = j + prev_mark_index
                if val in df['sam_th'].iloc[i + repeat:].values and jj not in ipre:
                    if df['is_saturated'].iloc[jj]:
                        print(f"Saturation found in pre-change stitch point, dropping index: {jj}")
                    else:
                        ipre.append(jj)

            for j, val in enumerate(df['sam_th'].iloc[i+repeat:]):
                jj = j+i+repeat
                if df['mark'].iloc[jj]:
                    break # If we hit the next stitch point, cut the search early
                if val in df['sam_th'].iloc[ipre].values and jj not in ipost:
                    if df['is_saturated'].iloc[jj]:
                        print(f"Saturation found in post-change stitch point, dropping index: {jj}")
                    else:
                        ipost.append(jj)
                   
            #    drop_points = []
            #    maxcounts = int(df['zinged_image'].iloc[j].max())
            #    if np.abs(maxcounts - 65535) < 2:
            #        print(f"Saturation found in pre-change stitch point, dropping index: {j}")
            #        drop_points.append(j)
            #ipre = [index for index in ipre if index not in drop_points]
            #for j in ipost:
            #    drop_points = []
            #    maxcounts = int(df['zinged_image'].iloc[j].max())
            #    if np.abs(maxcounts - 65535) < 2:
            #        print(f"Saturation found in post-change stitch point, dropping index: {j}")
            #        drop_points.append(j)
            #ipost = [index for index in ipost if index not in drop_points]
                
            stitch_pre = df[['sam_th', 'R']].iloc[ipre].loc[df['counts_ratio']>self.process_vars['stitch_cutoff']]
            stitch_post = df[['sam_th', 'R']].iloc[ipost].loc[df['counts_ratio']>self.process_vars['stitch_cutoff']]
            # Verify that the sam_th values are the same
            safe_stitch_values = list(set(stitch_pre['sam_th']) & set(stitch_post['sam_th']))
            # Record the number of values used for stitching
            #df.loc[igroup+i, 'safe_stitch_values'] = safe_stitch_values
            df.loc[igroup+i,'num_stitch_points'] = len(safe_stitch_values)
            # If no values were determined "SAFE" for stitching, 
            if len(safe_stitch_values) == 0 & self.process_vars['drop_failed_stitch']:
                print(f"----")
                print(f"Failed stitch occured at index {igroup+i}")
                print(f"Energy: {df['energy'].iloc[i]} eV")
                print(f"Theta: {df['sam_th'].iloc[i]}")
                print(f"Masking all non-stitched points")
                print(f"Check processing variables and re-calculate to try again")
                print(f"----")
                df.loc[igroup+i:,'failed_stitch_mask'] = 1
                break
            # We successfully found a stitch point and are moving on.
            if len(safe_stitch_values) == 1:
                warnings.warn(
                    f"The scan starting with index {df['fits_index'].iloc[0]} only has one stitch point."
                )

            # Proceed with stitching
            stitch_data = RealData(
                stitch_pre['R'].loc[stitch_pre['sam_th'].isin(safe_stitch_values)],
                stitch_post['R'].loc[stitch_post['sam_th'].isin(safe_stitch_values)]
            )
            stitch_ODR = ODR(stitch_data, linear, beta0=[1.0])
            stitch_output = stitch_ODR.run()
            scalei = stitch_output.beta[0]
            scale_erri = stitch_output.sd_beta[0]

            df.loc[igroup+i:,'scale'] = df['scale'].iloc[i:].apply(lambda x: x*scalei)
            df.loc[igroup+i:, 'scale_err'] = df.iloc[i:].apply(
                lambda x: x['scale']*((x['scale_err']/x['scale'])**2 + (scale_erri/scalei)**2)**0.5,
                axis=1
            )
            # Save this mark so we don't double-up in edge cases
            prev_mark_index = i
        return df