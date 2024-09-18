#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :imreg_dft.py
# @Time      :2022/10/9 16:54
# @Author    :kuisu_dgut@163.com

import math
import numpy as np
import scipy.ndimage.interpolation as ndii
import numpy.fft as fft
from . import utils


def translation(im0, im1, filter_pcorr=0, odds=1, constraints=None,
                reports=None):
    """
    Return translation vector to register images.
    It tells how to translate the im1 to get im0.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second (subject) image
        filter_pcorr (int): Radius of the minimum spectrum filter
            for translation detection, use the filter when detection fails.
            Values > 3 are likely not useful.
        constraints (dict or None): Specify preference of seeked values.
            For more detailed documentation, refer to :func:`similarity`.
            The only difference is that here, only keys ``tx`` and/or ``ty``
            (i.e. both or any of them or none of them) are used.
        odds (float): The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
            The value 1 is neutral, the converse of 2 is 1 / 2 etc.

    Returns:
        dict: Contains following keys: ``angle``, ``tvec`` (Y, X),
            and ``success``.
    """
    angle = 0
    report_one = report_two = None
    if reports is not None and reports.show("translation"):
        report_one = reports.copy_empty()
        report_two = reports.copy_empty()

    # We estimate translation for the original image...
    tvec, succ, confi_mask = _translation(im0, im1, filter_pcorr, constraints, report_one)

    ret = dict(tvec=tvec, success=succ, angle=angle, confi_mask=confi_mask)
    return ret


def _translation(im0, im1, filter_pcorr=0, constraints=None, reports=None):
    """
    The plain wrapper for translation phase correlation, no big deal.
    """
    # Apodization and pcorr don't play along
    # im0, im1 = [utils._apodize(im, ratio=1) for im in (im0, im1)]
    ret, succ, confi_mask = _phase_correlation(
        im0, im1,
        utils.argmax_translation, filter_pcorr, constraints, reports)
    return ret, succ, confi_mask


def _phase_correlation(im0, im1, callback, *args):
    """
    Computes phase correlation between im0 and im1

    Args:
        im0
        im1
        callback (function): Process the cross-power spectrum (i.e. choose
            coordinates of the best element, usually of the highest one).
            Defaults to :func:`imreg_dft.utils.argmax2D`

    Returns:
        tuple: The translation vector (Y, X). Translation vector of (0, 0)
            means that the two images match.
    """
    # TODO: Implement some form of high-pass filtering of PHASE correlation
    f0, f1 = [fft.fft2(arr) for arr in (im0, im1)]
    # spectrum can be filtered (already),
    # so we have to take precaution against dividing by 0
    eps = abs(f1).max() * 1e-15
    # cps == cross-power spectrum of im0 and im1
    cps = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
    # scps = shifted cps
    scps = fft.fftshift(cps)

    (t0, t1), success = callback(scps, *args)
    confi_mask = scps[int(t0) - 25:int(t0) + 25, int(t1) - 25:int(t1) + 25]
    # if success<0.02:
    #     import cv2
    #     scps = fft.fftshift(cps)
    #     win = cv2.createHanningWindow((9, 9), cv2.CV_32F)
    #     # scps = cv2.GaussianBlur(scps, (7,7), 1)
    #     scps = cv2.filter2D(scps,-1,win)
    #     (t0, t1), success = callback(scps, *args)
    ret = np.array((t0, t1))

    # _compensate_fftshift is not appropriate here, this is OK.
    t0 -= f0.shape[0] // 2
    t1 -= f0.shape[1] // 2

    ret -= np.array(f0.shape, int) // 2
    return ret, success, confi_mask


def _get_pcorr_shape(shape):
    ret = (int(max(shape) * 1.0),) * 2
    return ret


def _logpolar_filter(shape):
    """
    Make a radial cosine filter for the logpolar transform.
    This filter suppresses low frequencies and completely removes
    the zero freq.
    """
    yy = np.linspace(- np.pi / 2., np.pi / 2., shape[0])[:, np.newaxis]
    xx = np.linspace(- np.pi / 2., np.pi / 2., shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt(yy ** 2 + xx ** 2)
    filt = 1.0 - np.cos(rads) ** 2
    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[np.abs(rads) > np.pi / 2] = 1
    return filt


EXCESS_CONST = 1.1


def _get_log_base(shape, new_r):
    """
    Basically common functionality of :func:`_logpolar`
    and :func:`_get_ang_scale`

    This value can be considered fixed, if you want to mess with the logpolar
    transform, mess with the shape.

    Args:
        shape: Shape of the original image.
        new_r (float): The r-size of the log-polar transform array dimension.

    Returns:
        float: Base of the log-polar transform.
        The following holds:
        :math:`log\_base = \exp( \ln [ \mathit{spectrum\_dim} ] / \mathit{loglpolar\_scale\_dim} )`,
        or the equivalent :math:`log\_base^{\mathit{loglpolar\_scale\_dim}} = \mathit{spectrum\_dim}`.
    """
    # The highest radius we have to accomodate is 'old_r',
    # However, we cut some parts out as only a thin part of the spectra has
    # these high frequencies
    old_r = shape[0] * EXCESS_CONST
    # We are radius, so we divide the diameter by two.
    old_r /= 2.0
    # we have at most 'new_r' of space.
    log_base = np.exp(np.log(old_r) / new_r)
    return log_base


def _logpolar(image, shape, log_base, bgval=None):
    """
    Return log-polar transformed image
    Takes into account anisotropicity of the freq spectrum
    of rectangular images

    Args:
        image: The image to be transformed
        shape: Shape of the transformed image
        log_base: Parameter of the transformation, get it via
            :func:`_get_log_base`
        bgval: The backround value. If None, use minimum of the image.

    Returns:
        The transformed image
    """
    if bgval is None:
        bgval = np.percentile(image, 1)
    imshape = np.array(image.shape)
    center = imshape[0] / 2.0, imshape[1] / 2.0
    # 0 .. pi = only half of the spectrum is used
    theta = utils._get_angles(shape)
    radius_x = utils._get_lograd(shape, log_base)
    radius_y = radius_x.copy()
    ellipse_coef = imshape[0] / float(imshape[1])
    # We have to acknowledge that the frequency spectrum can be deformed
    # if the image aspect ratio is not 1.0
    # The image is x-thin, so we acknowledge that the frequency spectra
    # scale in x is shrunk.
    radius_x /= ellipse_coef

    y = radius_y * np.sin(theta) + center[0]
    x = radius_x * np.cos(theta) + center[1]
    output = np.empty_like(y)
    ndii.map_coordinates(image, [y, x], output=output, order=3,
                         mode="constant", cval=bgval)
    return output


def _get_ang_scale(ims, bgval, exponent='inf', constraints=None, reports=None):
    """
    Given two images, return their scale and angle difference.

    Args:
        ims (2-tuple-like of 2D ndarrays): The images
        bgval: We also pad here in the :func:`map_coordinates`
        exponent (float or 'inf'): The exponent stuff, see :func:`similarity`
        constraints (dict, optional)
        reports (optional)

    Returns:
        tuple: Scale, angle. Describes the relationship of
        the subject image to the first one.
    """
    assert len(ims) == 2, \
        "Only two images are supported as input"
    shape = ims[0].shape

    ims_apod = [utils._apodize(im) for im in ims]
    dfts = [fft.fftshift(fft.fft2(im)) for im in ims_apod]
    filt = _logpolar_filter(shape)
    dfts = [dft * filt for dft in dfts]

    # High-pass filtering used to be here, but we have moved it to a higher
    # level interface

    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    stuffs = [_logpolar(np.abs(dft), pcorr_shape, log_base)
              for dft in dfts]

    (arg_ang, arg_rad), success, confi_mask = _phase_correlation(
        stuffs[0], stuffs[1],
        utils.argmax_angscale, log_base, exponent, constraints, reports)

    angle = -np.pi * arg_ang / float(pcorr_shape[0])
    angle = np.rad2deg(angle)
    angle = utils.wrap_angle(angle, 360)
    scale = log_base ** arg_rad

    angle = - angle
    scale = 1.0 / scale

    if reports is not None:
        reports["shape"] = filt.shape
        reports["base"] = log_base

        if reports.show("spectra"):
            reports["dfts_filt"] = dfts
        if reports.show("inputs"):
            reports["ims_filt"] = [fft.ifft2(np.fft.ifftshift(dft))
                                   for dft in dfts]
        if reports.show("logpolar"):
            reports["logpolars"] = stuffs

        if reports.show("scale_angle"):
            reports["amas-result-raw"] = (arg_ang, arg_rad)
            reports["amas-result"] = (scale, angle)
            reports["amas-success"] = success
            extent_el = pcorr_shape[1] / 2.0
            reports["amas-extent"] = (
                log_base ** (-extent_el), log_base ** extent_el,
                -90, 90
            )

    if not 0.5 < scale < 2:
        raise ValueError(
            "Images are not compatible. Scale change %g too big to be true."
            % scale)

    return scale, angle


def transform_img(img, scale=1.0, angle=0.0, tvec=(0, 0),
                  mode="constant", bgval=None, order=1):
    """
    Return translation vector to register images.

    Args:
        img (2D or 3D numpy array): What will be transformed.
            If a 3D array is passed, it is treated in a manner in which RGB
            images are supposed to be handled - i.e. assume that coordinates
            are (Y, X, channels).
        scale (float): The scale factor (scale > 1.0 means zooming in)
        angle (float): Degrees of rotation (clock-wise)
        tvec (2-tuple): Pixel translation vector, Y and X component.
        mode (string): The transformation mode (refer to e.g.
            :func:`scipy.ndimage.shift` and its kwarg ``mode``).
        bgval (float): Shade of the background (filling during transformations)
            If None is passed, :func:`imreg_dft.utils.get_borderval` with
            radius of 5 is used to get it.
        order (int): Order of approximation (when doing transformations). 1 =
            linear, 3 = cubic etc. Linear works surprisingly well.

    Returns:
        np.ndarray: The transformed img, may have another
        i.e. (bigger) shape than the source.
    """
    if img.ndim == 3:
        # A bloody painful special case of RGB images
        ret = np.empty_like(img)
        for idx in range(img.shape[2]):
            sli = (slice(None), slice(None), idx)
            ret[sli] = transform_img(img[sli], scale, angle, tvec,
                                     mode, bgval, order)
        return ret

    if bgval is None:
        bgval = utils.get_borderval(img)

    bigshape = np.round(np.array(img.shape) * 1.2).astype(int)
    bg = np.zeros(bigshape, img.dtype) + bgval

    dest0 = utils.embed_to(bg, img.copy())
    if scale != 1.0:
        dest0 = ndii.zoom(dest0, scale, order=order, mode=mode, cval=bgval)
    if angle != 0.0:
        dest0 = ndii.rotate(dest0, angle, order=order, mode=mode, cval=bgval, reshape=False)

    if tvec[0] != 0 or tvec[1] != 0:
        dest0 = ndii.shift(dest0, tvec, order=order, mode=mode, cval=bgval)

    bg = np.zeros_like(img) + bgval
    dest = utils.embed_to(bg, dest0)
    return dest


def _get_odds(angle, target, stdev):
    """
    Determine whether we are more likely to choose the angle, or angle + 180°

    Args:
        angle (float, degrees): The base angle.
        target (float, degrees): The angle we think is the right one.
            Typically, we take this from constraints.
        stdev (float, degrees): The relevance of the target value.
            Also typically taken from constraints.

    Return:
        float: The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
    """
    ret = 1
    if stdev is not None:
        diffs = [abs(utils.wrap_angle(ang, 360))
                 for ang in (target - angle, target - angle + 180)]
        odds0, odds1 = 0, 0
        if stdev > 0:
            odds0, odds1 = [np.exp(- diff ** 2 / stdev ** 2) for diff in diffs]
        if odds0 == 0 and odds1 > 0:
            # -1 is treated as infinity in _translation
            ret = -1
        elif stdev == 0 or (odds0 == 0 and odds1 == 0):
            ret = -1
            if diffs[0] < diffs[1]:
                ret = 0
        else:
            ret = odds1 / odds0
    return ret

def _get_precision(shape, scale=1):
    """
    Given the parameters of the log-polar transform, get width of the interval
    where the correct values are.

    Args:
        shape (tuple): Shape of images
        scale (float): The scale difference (precision varies)
    """
    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    # * 0.5 <= max deviation is half of the step
    # * 0.25 <= we got subpixel precision now and 0.5 / 2 == 0.25
    # sccale: Scale deviation depends on the scale value
    Dscale = scale * (log_base - 1) * 0.25
    # angle: Angle deviation is constant
    Dangle = 180.0 / pcorr_shape[0] * 0.25
    return Dangle, Dscale


def similarity(im0, im1, numiter=1, order=3, constraints=None,
                filter_pcorr=0, exponent='inf', bgval=None, reports=None):
    """
    This function takes some input and returns mutual rotation, scale
    and translation.
    It does these things during the process:

    * Handles correct constraints handling (defaults etc.).
    * Performs angle-scale determination iteratively.
      This involves keeping constraints in sync.
    * Performs translation determination.
    * Calculates precision.

    Returns:
        Dictionary with results.
    """
    if bgval is None:
        bgval = utils.get_borderval(im1, 5)

    shape = im0.shape
    if shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif im0.ndim != 2:
        raise ValueError("Images must be 2-dimensional.")

    # We are going to iterate and precise scale and angle estimates
    scale = 1.0
    angle = 0.0
    im2 = im1

    constraints_default = dict(angle=[0, None], scale=[1, None])
    if constraints is None:
        constraints = constraints_default

    # We guard against case when caller passes only one constraint key.
    # Now, the provided ones just replace defaults.
    constraints_default.update(constraints)
    constraints = constraints_default

    # During iterations, we have to work with constraints too.
    # So we make the copy in order to leave the original intact
    constraints_dynamic = constraints.copy()
    constraints_dynamic["scale"] = list(constraints["scale"])
    constraints_dynamic["angle"] = list(constraints["angle"])

    if reports is not None and reports.show("transformed"):
        reports["after_tform"] = [im2.copy()]

    for ii in range(numiter):
        newscale, newangle = _get_ang_scale([im0, im2], bgval, exponent,
                                            constraints_dynamic, reports)
        scale *= newscale
        angle += newangle

        constraints_dynamic["scale"][0] /= newscale
        constraints_dynamic["angle"][0] -= newangle

        im2 = transform_img(im1, scale, angle, bgval=bgval, order=order)

    # Here we look how is the turn-180
    target, stdev = constraints.get("angle", (0, None))
    odds = _get_odds(angle, target, stdev)

    # now we can use pcorr to guess the translation
    res = translation(im0, im2, filter_pcorr, odds,
                      constraints, reports)

    # The log-polar transform may have got the angle wrong by 180 degrees.
    # The phase correlation can help us to correct that
    angle += res["angle"]
    res["angle"] = utils.wrap_angle(angle, 360)

    # don't know what it does, but it alters the scale a little bit
    # scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    Dangle, Dscale = _get_precision(shape, scale)

    res["scale"] = scale
    res["Dscale"] = Dscale
    res["Dangle"] = Dangle
    # 0.25 because we go subpixel now
    res["Dt"] = 0.25

    return res


if __name__ == "__main__":
    pass
