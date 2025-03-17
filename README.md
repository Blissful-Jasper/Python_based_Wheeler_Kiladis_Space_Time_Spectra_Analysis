# wkSpaceTime
A Python code for the construction of the Wheeler-Kiladis Space-Time Spectra.


This code is based on the functions provided in the following GitHub repositories:

- [wavenumber_frequency_functions.py](https://github.com/Blissful-Jasper/wavenumber_frequency/blob/master/wavenumber_frequency_functions.py)

- [wk_analysis.py](https://github.com/Blissful-Jasper/wk_spectra/blob/master/wk_spectra/wk_analysis.py)

- [spectrum.py](https://github.com/Blissful-Jasper/mcclimate/blob/master/spectrum.py)



- The adjustments made to the code focus on improving its computational speed. about 2 seconds

- While the results might slightly differ from those generated by the official NCL website or the original implementations in the aforementioned repositories, the extraction of symmetric and antisymmetric signals from the overall wave fluctuations is acceptable when compared to the NCL results.

- This version is a straightforward implementation, without extensive encapsulation, providing a good foundation for understanding the calculation process of the space-time spectrum. 

- The data used in this code can be obtained from the NOAA's Uninterpolated OLR Data, with daily mean data from 1997 to 2014 used in this code.

- [data used from here](https://psl.noaa.gov/data/gridded/data.uninterp_OLR.html)

## this code output
the code used here is：Python_wk_spacetime.ipynb

![image](https://github.com/user-attachments/assets/690df297-d140-460d-8ce8-bca411bb10d7)

## same data from ncl outpuy

the code used here is：wk_time4.ncl

- https://www.ncl.ucar.edu/Document/Functions/Diagnostics/wkSpaceTime.shtml
  
![image](https://github.com/user-attachments/assets/e2118483-efd7-4033-b3f8-58440476a6e5)
