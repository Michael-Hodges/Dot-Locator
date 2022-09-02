import numpy as np
import cv2 as cv
from typing import Optional
import unittest

def _knn_filter(centers):
    KNN_4 = np.zeros(centers.shape[0])
    idx = 0
    for x_in, y_in in centers:
        # 4 needed since corners have 3 neighbors plus themselves (length = 0)
        val = np.inf*np.ones(4)
        for x_comp, y_comp in centers:
            dist = np.sqrt((x_in-x_comp)**2 + (y_in-y_comp)**2)
            if (dist < np.max(val)):
                val[np.argmax(val)] = dist
        KNN_4[idx] = np.mean(val)
        idx += 1
    return (KNN_4 < (KNN_4.mean()+2*KNN_4.std()))

def locate_dots(file: str = 'dots.png', outlier_removal: bool = True,
        scale: Optional[float] = None, im_write: bool = False, 
        output_fmt: Optional[str] = None) -> np.array(np.float64):
    """Locates the enter of dots in an image. Using Harris detector, contours,
       min enclosing circles, and optional outlier removal.

        Parameters:
            file:            path to image
            outlier_removal: optional knn filtering based on distance
            scale:           pixel to measurement unit scale
            im_write:        write dots on image
            output_fmt:      Format of output file <'csv' | 'txt'>

        Returns:
            Centers: Numpy array of center of dots as floats
    """
    img = cv.imread(file)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    blank_image = np.zeros_like(img , dtype = np.uint8)

    # Set background to white for use with cv.findContours
    blank_image[dst>0.01*dst.max()]=[255,255,255]
    blank_image = cv.cvtColor(blank_image, cv.COLOR_BGR2GRAY)

    contours, _ = cv.findContours(blank_image, cv.RETR_TREE, 
                                  cv.CHAIN_APPROX_NONE)

    centers = []
    rad = []
    for cnt in contours:
        (x,y),radius = cv.minEnclosingCircle(cnt)
        centers.append((x,y))
        rad.append(radius)

    centers = np.array(centers)
    rad = np.array(rad)

    if outlier_removal:
        idx = _knn_filter(centers)
        centers = centers[idx]
        rad = rad[idx]

    file_prefix = file.split('.')[0]
    if im_write:
        for center, rad in zip(centers, rad):
            x,y = center
            center = (int(x), int(y))
            rad = int(rad)
            cv.circle(img, center, rad, (0,255,0),1)
        cv.imwrite(''.join([file_prefix, "_located.png"]), img)
    if scale:
        centers = scale*centers

    if output_fmt:
        if output_fmt == 'txt':
            np.savetxt(''.join([file_prefix, "_located.", output_fmt]), 
                       centers, delimiter='\t')
        if output_fmt == 'csv':
            np.savetxt(''.join([file_prefix, "_located.", output_fmt]), 
                       centers, delimiter=',')
    return centers


class TestDotLocator(unittest.TestCase):
    def setUp(self,):
        self.files = ['./dots.png']
        self.lengths = [676]

    def test_count(self,):
        for file, length in zip(self.files, self.lengths):
            ans = locate_dots(file=file)
            self.assertEqual(len(ans), length)

    def test_albus(self,):
        for file in self.files:
            test_centers = locate_dots(file=file)
            correct_centers = np.loadtxt(''.join(['./test_files/',
                                         (file.split('.')[1]).split('/')[1], 
                                         '_located.csv']), delimiter=',')
            np.testing.assert_allclose(correct_centers, test_centers, 
                                        rtol=0, atol=0.02)
    def test_bellatrix(self,):
        for file in self.files:
            test_centers = locate_dots(file=file)
            correct_centers = np.loadtxt(''.join(['./test_files/',
                                         (file.split('.')[1]).split('/')[1], 
                                         '_located.csv']), delimiter=',')
            np.testing.assert_allclose(correct_centers, test_centers, 
                                        rtol=0, atol=0.1)
    def test_cedrix(self,):
        for file in self.files:
            test_centers = locate_dots(file=file)
            correct_centers = np.loadtxt(''.join(['./test_files/',
                                         (file.split('.')[1]).split('/')[1], 
                                         '_located.csv']), delimiter=',')
            np.testing.assert_allclose(correct_centers, test_centers, 
                                        rtol=0, atol=0.2)

if __name__ == '__main__':
    locate_dots(file = 'dots.png', outlier_removal = True, im_write = True, 
                output_fmt = 'csv')
