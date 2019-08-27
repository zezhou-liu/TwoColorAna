# Manual of the Two Color data Analysis module(TCA)
  This module is intended to contain various analysis functions to help process the data. The pipeline of working is shown here:
  
  [under construction]
  
  If you have any questions, please send an email to zezhou_liu@outlook.com directly.

## Introduction to TCA
  The scripts provided here are based on ``quick and dirty" principle. UI is not provided. However, please let me know if you want to help to improve the usr experience.

## Input data requirement
  Assuming the position information has been extract from the raw data, the data has to be arranged in the following hierachy:
  
  [under construction]
  
## Module
### bashload(main_path)
Load ALL the data inside the "main_path". This function returns a dictionary containing all the relavent data. The name format is: 'eccentricity_videoclip_y1x'. For example, 'ecc03_2_y1x' means the data corresponding to /2019mmdd_ecc03/2/y1x.txt.


Example of use:

  import module
  
  main_path = 'C:/Users/admin'
  
  tot_file = module.bashload(main_path)
  
  print(tot_file['ecc03_2_y1x'])
 
### 
