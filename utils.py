"""Utility functions for CS194-26 Project 1"""

from typing import Callable, Tuple

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktrans
from skimage.feature import canny


def convert_to_bgr_channels(
    *, filepath: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three arrays of b, g, r channels

    Parameters:
        filepath (str): Name of file

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]): R, G, B color channels

    """
    # read in the image
    im = skio.imread(filepath)

    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height : 2 * height]
    r = im[2 * height : 3 * height]

    return b, g, r


def vertical_translation(
    *,
    im: np.ndarray,
    offset: int,
    fill_color: int = 0,
    show: bool = False,
) -> np.ndarray:
    """Shift the input image `im` vertically by `offset` number of pixels.

    Parameters:
        im (np.ndarray): Image represented by 2D matrix
        offset (int): The number of pixels to translate the image
        fill_color (int): Pixel value to fill in lost pixels
        show (bool): Option to show translated image

    Returns:
        (np.ndarray): The translated image in matrix form
    """
    if not offset:
        return im

    width, height = im.shape
    positive_shift = offset > 0

    mask = np.full(shape=(abs(offset), height), fill_value=fill_color)
    spliced_im = im[offset:, :] if positive_shift else im[:offset, :]
    shifted = [spliced_im, mask] if positive_shift else [mask, spliced_im]
    combined = np.vstack(shifted)

    if show:
        skio.imshow(combined)
    return combined


def horizontal_translation(
    *,
    im: np.ndarray,
    offset: int,
    fill_color: int = 0,
    show: bool = False,
) -> np.ndarray:
    """Shift the input image `im` horizontally by `offset` number of pixels.

    Parameters:
        im (np.ndarray): Image represented by 2D matrix
        offset (int): The number of pixels to translate the image
        fill_color (int): Pixel value to fill in lost pixels
        show (bool): Option to show translated image

    Returns:
        (np.ndarray): The translated image in matrix form
    """
    if not offset:
        return im

    width, height = im.shape
    positive_shift = offset > 0

    mask = np.full(shape=(width, abs(offset)), fill_value=fill_color)
    spliced_im = im[:, offset:] if positive_shift else im[:, :offset]
    shifted = [mask, spliced_im] if positive_shift else [spliced_im, mask]
    combined = np.hstack(shifted)

    if show:
        skio.imshow(combined)
    return combined


def shift_image(
    *,
    im: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> np.ndarray:
    """Shift the input image `im` horizontally and vertically.

    Parameters:
        im (np.ndarray): Image represented by 2D matrix
        x_offset (int): The number of pixels to shift image in x direction
        y_offset (int): The number of pixels to shift image in y direction
    """
    v_im = np.roll(im, shift=y_offset, axis=0)
    return np.roll(v_im, shift=x_offset, axis=1)


def ssd(
    *,
    im1: np.ndarray,
    im2: np.ndarray,
    alpha: float = 0.2,
):
    """Returns the Sum of Squared Distances between `im1` and `im2`.

    Parameters:
        im1 (np.ndarray): Image represented by 2D matrix
        im2 (np.ndarray): Image represented by 2D matrix
        alpha (float): Percentage of image to cut off each side

    Returns:
        (float): SSD metric between input images
    """

    # Make sure arrays have same shape
    assert im1.shape == im2.shape

    x_cut, y_cut = (int(val * alpha) for val in im1.shape)
    im1_reduced = im1[x_cut:-x_cut, y_cut:-y_cut]
    im2_reduced = im2[x_cut:-x_cut, y_cut:-y_cut]

    return np.sum(np.sum(np.square(im1_reduced - im2_reduced)))


def ssd_edge_detection(
    *,
    im1: np.ndarray,
    im2: np.ndarray,
    alpha: float = 0.2,
):
    """Returns the Sum of Squared Distances between `im1` and `im2`.

    Parameters:
        im1 (np.ndarray): Image represented by 2D matrix
        im2 (np.ndarray): Image represented by 2D matrix
        alpha (float): Percentage of image to cut off each side

    Returns:
        (float): SSD metric between input images
    """

    # Make sure arrays have same shape
    assert im1.shape == im2.shape

    x_cut, y_cut = (int(val * alpha) for val in im1.shape)
    im1_reduced = im1[x_cut:-x_cut, y_cut:-y_cut]
    im2_reduced = im2[x_cut:-x_cut, y_cut:-y_cut]

    return np.sum(np.sum(np.square(im1_reduced ^ im2_reduced)))


def align_to_base(
    *,
    im: np.ndarray,
    base_im: np.ndarray,
    max_displacement: int = 15,
    loss_f: Callable = ssd,
) -> Tuple[float, np.ndarray, Tuple[float, float]]:
    """Aligns an input image to the base image
    Parameters:
        im (np.ndarray): Image represented by 2D matrix
        base_im (np.ndarray): Base image represented by 2D matrix to map
                        ``im`` to.
        max_displacement (int): The maximum number of pixels to try shifting
        loss_f (Callable): Function use to measure distance

    Returns:
        (error, aligned image, displacement): Tuple representing the alignment
    """
    optimal_err, optimal_im, optimal_shift = float("inf"), None, None

    for x_disp in range(-max_displacement, max_displacement + 1):
        for y_disp in range(-max_displacement, max_displacement + 1):
            candidate_im = np.roll(im, shift=y_disp, axis=0)
            candidate_im = np.roll(candidate_im, shift=x_disp, axis=1)
            candidate_error = loss_f(im1=base_im, im2=candidate_im)
            if candidate_error < optimal_err:
                optimal_err = candidate_error
                optimal_im = candidate_im
                optimal_shift = (x_disp, y_disp)

    return optimal_err, optimal_im, optimal_shift


def align_pyramid(
    *,
    im: np.ndarray,
    base_im: np.ndarray,
    depth: int = 5,
    max_displacement: int = 15,
    loss_f: Callable = ssd,
    edge_detection: bool = False,
    x_shift: int = 0,
    y_shift: int = 0,
) -> np.ndarray:
    """Uses an image pyramid to align image channel to base
    Parameters:
        im (np.ndarray): Image represented by 2D matrix
        base_im (np.ndarray): Base image represented by 2D matrix to map
                        ``im`` to.
        depth (int): The number of levels for the image pyramid
        max_displacement (int): The maximum number of pixels to try shifting
        loss_f (Callable): Function use to measure distance
        edge_detection (bool): Option to use edge detection for alignment
        x_shift (int): Current shift of pixels on `im` in x-direction
        y_shift (int): Current shift of pixels on `im` in y-direction

    Returns:
        (np.ndarray): Aligned image channel as a 2D matrix
    """
    if not depth:
        return im, x_shift, y_shift

    scale_factor = 2 ** (depth - 1)

    reduced_im = sktrans.rescale(im, 1 / scale_factor)
    reduced_base_im = sktrans.rescale(base_im, 1 / scale_factor)

    if edge_detection:
        im_edges = canny(reduced_im, sigma=3)
        base_edges = canny(reduced_base_im, sigma=3)

        _, _, (x_delta, y_delta) = align_to_base(
            im=im_edges,
            base_im=base_edges,
            max_displacement=max_displacement,
            loss_f=ssd_edge_detection,
        )
    else:
        _, _, (x_delta, y_delta) = align_to_base(
            im=reduced_im,
            base_im=reduced_base_im,
            max_displacement=max_displacement,
            loss_f=loss_f,
        )

    im = shift_image(
        im=im,
        x_offset=(x_delta * scale_factor),
        y_offset=(y_delta * scale_factor),
    )

    return align_pyramid(
        im=im,
        base_im=base_im,
        depth=depth - 1,
        max_displacement=max_displacement,
        loss_f=loss_f,
        edge_detection=edge_detection,
        x_shift=x_shift + (x_delta * scale_factor),
        y_shift=y_shift + (y_delta * scale_factor),
    )


def align_full_image(
    *,
    b: np.ndarray,
    g: np.ndarray,
    r: np.ndarray,
    depth: int = 5,
    max_displacement: int = 15,
    loss_f: Callable = ssd,
    edge_detection: bool = False,
) -> np.ndarray:
    """Aligns green and red channels to blue channel

    Parameters:
        b (np.ndarray): Blue color channel represented by 2D matrix
        g (np.ndarray): Green color channel represented by 2D matrix
        r (np.ndarray): Red color channel represented by 2D matrix
        depth (int): The number of levels for the image pyramid
        max_displacement (int): The maximum number of pixels to try shifting
        loss_f (Callable): Function use to measure distance
        edge_detection (bool): Option to use edge detection for alignment
    """
    ag, g_x_shift, g_y_shift = align_pyramid(
        im=g,
        base_im=b,
        max_displacement=max_displacement,
        edge_detection=edge_detection,
        depth=depth,
        loss_f=loss_f,
    )

    ar, r_x_shift, r_y_shift = align_pyramid(
        im=r,
        base_im=b,
        max_displacement=max_displacement,
        edge_detection=edge_detection,
        depth=depth,
        loss_f=loss_f,
    )

    # create a color image
    return (
        np.dstack([ar, ag, b]),
        (g_x_shift, g_y_shift),
        (r_x_shift, r_y_shift),
    )
