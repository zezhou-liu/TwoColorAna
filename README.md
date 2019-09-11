# Manual of the Two Color data Analysis module(TCA)
  This module is intended to contain various analysis functions to help process the data. The pipeline of working is shown here:
  
  [under construction]
  
  If you have any questions, please send an email to zezhou_liu@outlook.com directly.

## Introduction to TCA
  The scripts provided here are based on ``quick and dirty" principle. UI is not provided. However, please let me know if you want to help to improve the usr experience.

## Input data requirement
  Assuming the position information has been extract from the raw data, the data has to be arranged in the following hierachy:
  
  [under construction]
  
  Please notice that you are ONLY allowed to add .txt file in folders(except the top folders) to help explain your data.
## Module
### bashload(main_path)
**Input**: Root path. For example, if you save your data in '../main/20180801_ecc0/1/y1x.txt', the main_path is '../main'. No '/' in the end.

**Output**: Data handle, a dictionary containing all the data. Item naming format: **'ecc03_1_y1x'**.

Load ALL the data inside the "main_path". This function returns a dictionary containing all the relavent data. The name format is: 'eccentricity_videoclip_y1x'. For example, 'ecc03_2_y1x' means the data corresponding to /2019mmdd_ecc03/2/y1x.txt.

Example of use:
```python
  import module 
  main_path = 'C:/Users/admin'
  handle, tot_file = module.bashload(main_path)
  print(handle.tot_file['ecc03_2_y1x'])
```
### bashvector(handle)
**Input**: Data handle.

**Output**: Data handle, a dictionary containing all the vectors. Item naming format: **'ecc03_1_delx'**.

Suppose that the position of DNA is (x1, y1) and (x2, y2) for YOYO-1 and YOYO-3 channel respectively. This function returns the vector (x2-x1, y2-y1). 

Example of use:
```python
  import module 
  main_path = 'C:/Users/admin'
  handle, tot_file = module.bashload(main_path)
  handle, tot_vector = module.bashvector(handle)
  print(handle.tot_vector['ecc03_1_delx'])
```
### bashoverlay(handle)
**Input**: Data handle.

**Output**: Data handle, a dictionary containing all the vectors after merging. Item naming format: **'ecc03_delx'**.

This function merges all the videos with the same eccentricity into one dictionary item. The data handle must contain *tot_vector* before processing.

Example of use:
```python
  import module 
  main_path = 'C:/Users/admin'
  handle, tot_file = module.bashload(main_path)
  handle, tot_vector = module.bashvector(handle)
  handle, tot_vec_overlay = module.bashvector(handle)
  print(handle.tot_vec_overlay['ecc03_delx'])
```
