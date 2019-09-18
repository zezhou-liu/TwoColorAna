# Manual of the Two Color data Analysis module(TCA)
  This module is intended to contain various analysis functions to help process the data. The pipeline of working is shown here:
  
  [under construction]

## Introduction to TCA
  The scripts provided here are based on ``quick and dirty" principle. You might need to tweak the package a bit to suit your need. UI is not provided. However, please let me know if you want to help to improve the usr experience.

## Input data requirement
  Assuming the position information has been extract from the raw data, the data has to be arranged in the following hierachy:
  
  [under construction]
  
  Please notice that you are ONLY allowed to add .txt file in folders(except the top folders) to help explain your data.
## Module

### bashload(main_path)
Load ALL the data inside the "main_path". This function returns a dictionary containing all the relavent data. The name format is: 'eccentricity_videoclip_y1x'. For example, 'ecc03_2_y1x' means the data corresponding to /2019mmdd_ecc03/2/y1x.txt.
**Input**: Root path. For example, if you save your data in '../main/20180801_ecc0/1/y1x.txt', the main_path is '../main'. No '/' in the end.

**Output**: Data handle, a dictionary containing all the data. Item naming format: **'ecc03_1_y1x'**.

### bashvector(handle, mode='raw')
Suppose that the position of DNA is (x1, y1) and (x2, y2) for YOYO-1 and YOYO-3 channel respectively. 
This function returns the vector (x2-x1, y2-y1). 
**Input**: Data handle. 

**Parameters**: 
mode: There are two modes of *bashvector* in current version. 'raw' mode will calculate vector based on raw data saved in *handle.tot_file*. 'clean' mode will calculate vector based on data whic has been cropped and shifted. The shifted data is saved in *handle.tot_file_shift*. 

**Output**: Data handle, A dictionary containing all the vectors. In 'raw' mode, the dictionary will be attached to handle as *handle.tot_vector*. In 'clean' mode, the dictionary will be attached to handle as *handle.tot_vector_clean*. In both modes, item naming format is : **'ecc03_1_delx'**.

### bashoverlay(handle, mode='raw', set = 'vector')
This function merges all the videos with the same eccentricity into one dictionary item. 

**Input**: Data handle.

**Parameters**
mode: There are two modes of *bashoverlay* in current version. 'raw' mode will calculate the overlay on raw data. 'clean' mode will calculate the overlay based on data which has been cropped and shifted.  

set: Set parameter controls the data you want to overlay. 'vector' set will calculate the overlay on position separation vector. 'position' set will calculate the overlay on absolute position. Here I recommand to use *modd='clean', set='position'* to calculate the position because the raw data contains noise generated from stage movement and cropping.

**Output**: Data handle, A dictionary containing all the data after merging. For set 'vector', the vector separation is saved in *handle.tot_vec_overlay* or *handle.tot_vec_overlay_clean* depending on the *mode*. For set 'position', the position overlay is saved in *handle.tot_pos_overlay* or *handle.tot_pos_overlay_shift* depending on the *mode*. Item naming format: For set 'vector', it's **'ecc03_delx'**. For set 'position', it's *'ecc03_y1x'*. 

| mode  | set      | dict name                    | item name  |
|-------|----------|------------------------------|------------|
| raw   | vector   | handle.tot_vec_overlay       | ecc03_delx |
| raw   | position | handle.tot_pos_overlay       | ecc03_y1x  |
| clean | vector   | handle.tot_vec_overlay_clean | ecc03_delx |
| clean | position | handle.tot_pos_overlay_clean | ecc03_y1x  |

### bashfree(handle)
Calculate the free energy landscape of the vector. PCA is applied to extract the It requires *handle.tot_vector* and *handle.tot_vec_overlay*. Please refer page 107 of [this thesis](https://pdfs.semanticscholar.org/bb60/688e23a2057fa7e27d12c9e29a3bfbe66264.pdf) for free energy calculation. This module is compatible with two identical chain experiment.

**Input**: Data handle.

**Output**: Data handle, Free energy landscape along different axis. All the free energy data is saved in *handle.tot_free* dictionary. Item naming format is : *ecc03_F* free energy along the principle axis. *ecc03_Fs* free energy along the second principle axis. *ecc03_bins* x-axis(position in pixel) of *ecc03_F*. *ecc03_binss* x-axis(position in pixel) of *ecc03_Fs*.

### bashroi(handle)
Help remove the bad data points outside ROI. The points outside ROI is caused by the nature of denoise module. Data points that are too dim usually locate at the upper right corner. This function will pop up a window and let usr select ROI by clicking the left-upper corner and right-lower corner of ROI. *handle.tot_vector* is ploted to help justification.

**Input**: Data handle.

**Output**: Data handle, Roi. roi is also attached to handle as *handle.roi* and *roi.json* file will be saved in *~/data* folder. 

### bashclean(handle)
Remove the frames outside ROI. It requires *~/data/roi.json* file which is generated by *bashroi*.

**Input**: Data handle.

**Output**: Data handle, tot_file_clean. Data set after frame removal is saved in *handle.tot_file_clean*. The mask file is saved in *handle.tot_roimask*. 

### bashshift(handle)
Shift the data to zero to allow absolute position tracking. It requires *handle.tot_file_clean* which is generated from *bashclean*. We use k-means to remove the stage shift noise during the experiment. The shift depends on y3 channel.

**Input**: Data handle.

**Parameters**: n_init, initiation times for the clustering; max_iter, max times of iteration for each init;tol, tol to stop each iteration.

**Output**: Data handle, tot_file_shift. The shift file is attached to handle as *handle.tot_file_shift*. Item naming format is *'ecc03_1_y1x'*.

Example of use:
```python
  import module 
  main_path = 'C:/Users/admin'
  handle, tot_file = module.bashload(main_path)
  handle, tot_vector = module.bashvector(handle)
  handle, tot_vec_overlay = module.bashvector(handle)
  print(handle.tot_vec_overlay['ecc03_delx'])
```
# Contact
If you have any questions, please send an email to zezhou_liu@outlook.com directly.