# Dot-Locator
Locates dots in an image and optionally filters outliers.

## Requirements
General requirements are as follows:
* Numpy
* OpenCV

A prepackaged conda environment can be loaded as follows:
```
conda env create -f environment.yml
```

## Usage
Intended usage to be done from within another script calling the main function.
More information is in the function docstring.
```
locate_dots(file: str = 'dots.png', outlier_removal: bool = True,
        scale: Optional[float] = None, im_write: bool = False, 
        output_fmt: Optional[str] = None) -> np.array(np.float64):
```
Example usage with a file 'img.png', drawing located dots on the image, scaling
each pixel to 0.1mm, and writing to a csv is as follows:
```
locate_dots(file='img.png', scale=0.1, im_write=True, output_fmt='csv')
```
## Methodology
### Overview
The function takes the following steps to find the dots.
* Harris Corner Detector
* Find Contours
* Minimum Enclosing circle for each contour found
* Filter outliers using K-Nearest Neighbors with distance

#### Accuracy
Visual inspection shows of the output image shows that the accuracy of the
current dot locator does not meet the specs provided (e.g. 0.02, 0.1, and 0.2
pixels). Using a subpixel edge detector as specified in 
[Automated calibration of multi-camera-projector structured light systems for volumetric high-speed 3D surface reconstructions](https://www.researchgate.net/figure/Image-processing-steps-to-detect-dots-a-d-Identification-of-the-printed-dots-captured_fig1_329463216)
would improve our results. However, our implementation provides the desired 
output forms of csv, tab deliminated text, and numpy array for each team.

### Outlier Detection
Outliers are removed based on the 3 Nearest Neigbors. This number was chosen
since corners have a max of 3 neighbors. In the implementation an array of 4
is populated since a value of 0.0 will be inside every array when compared
to itself. 

Examples are as follows:
* Outlier Removal:
![Outlier Removal](https://github.com/Michael-Hodges/Dot-Locator/dots_located.png?raw=true)
* No Outlier Removal:
![No Outlier Removal](https://github.com/Michael-Hodges/Dot-Locator/dots_unfiltered.png?raw=true)

## Running Tests
```
python -m unittest dot_locator.py
```
Files use for the test are located in the test_files directory.
Tests ran are as follows:
* Count: verifies the number of elements found is correct.
* Centers: Verify centers count are within x pixels per team
    * Albus: within 0.02 pixels
    * Bellatrix: within 0.1 pixels
    * Cedrix: within 0.2 pixels
