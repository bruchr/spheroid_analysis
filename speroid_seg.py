import numpy as np
import scipy.ndimage
from skimage.measure import label


def spheroid_seg(seg, repetitions=40, verbose=False):
    seg = seg>=1

    # se = skimage.morphology.octahedron(10)
    # for _ in range(repetitions):
    #     seg = skimage.morphology.binary_dilation(seg, se)
    # for _ in range(repetitions):
    #     seg = skimage.morphology.binary_erosion(seg, se)
    scipy.ndimage.binary_dilation(seg, iterations=repetitions, output=seg)
    scipy.ndimage.binary_erosion(seg, iterations=repetitions, border_value=1, output=seg)
    if verbose:
        print('Closing done')
        print(seg.shape)

    scipy.ndimage.binary_fill_holes(seg, output=seg)
    if verbose:
        print('Fill holes done')
        print(seg.shape)

    labels = label(seg)
    if labels.max() == 0:
        raise Exception('No labels in the image')
    seg = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    if verbose:
        print('Select biggest label done')
        print('Shape of segmentation: {}; dtype: {}'.format(seg.shape,seg.dtype))

    return seg.astype(np.uint8)


if __name__ == "__main__":
    import tifffile as tiff

    path_seg = 'path/to/instance/segmentation.tif'

    seg = tiff.imread(path_seg)

    seg = spheroid_seg(seg)

    # tiff.imwrite(path_seg.replace('.tif','_spheroid_.tif'), seg)