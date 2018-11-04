"""
This module provides frequency arithmetic for linear and quadratic responses.
"""

from enum import Enum
import numpy


class Subset(Enum):
    """
    Description of the subsets of interactive frequencies.
    """

    LINEAR = 0x000F
    """Linear frequencies :math:`f_i`"""
    DIAGONAL = 0x00F0
    """Diagonal (harmonic) frequencies :math:`2 f_i`"""
    PLUS = 0x0F00
    """Sum frequencies :math:`f_i + f_j`"""
    MINUS = 0xF000
    """Difference frequencies :math:`|f_i - f_j|`"""
    ALL = 0xFFFF
    """All frequencies, equivalent to LINEAR, QUADRATIC"""
    QUADRATIC = 0xFFF0
    """Quadratic frequencies, equivalent to DIAGONAL, PLUS, MINUS"""


def hertz2fick(dt, duration, frequency):
    """
    Convert a frequency given in hertz to the corresponding FFT index.
    Note that negative frequencies are converted to the right of positive frequencies.

    :param dt: sampling time (seconds)
    :type dt: float
    :param duration: experiment duration (seconds) such that :math:`1 / duration` represents the minimal frequency
    :type duration: float
    :param frequency: frequency (hertz) to convert
    :type frequency: float
    :return: index of the frequency in the FFT
    :rtype: float

    .. highlight:: python
    .. code-block:: python

        >>> qsa.domain.hertz2fick(0.001, 10, 1)
        10
        >>> qsa.domain.hertz2fick(0.001, 10, -1)
        9990
    """
    n = duration / dt if frequency < 0 else 0
    return int(round(n + frequency * duration))


def hertz2ficks(dt, duration, frequencies):
    """
    Convert an array of frequencies in hertz to the corresponding array of FFT indices.

    :param dt: sampling time (seconds)
    :type dt: float
    :param duration: experiment duration (seconds) such that :math:`1 / duration` represents the minimal frequency
    :type duration: float
    :param frequencies: array of frequencies (hertz) to convert
    :type frequencies: numpy.ndarray
    :return: indices of the frequencies in the FFT
    :rtype: numpy.ndarray

    .. highlight:: python
    .. code-block:: python

        >>> qsa.domain.hertz2ficks(0.001, 10, numpy.array([-10, -1, +1, +10]))
        [9900, 9990, 10, 100]
    """
    return list(map(
        lambda frequency: hertz2fick(dt, duration, frequency),
        frequencies))


def order(frequencies, symmetric=True, subset=Subset.ALL):
    """
    Compute first and second order interactive frequencies.

    :param frequencies: array of fundamental frequencies (hertz) ; they must be strictly positive
    :type frequencies: numpy.ndarray
    :param symmetric: True to generate both negative and positive frequencies ; False to generate positive frequencies only
    :type symmetric: bool
    :param subset: specify the frequency combinations (see also :py:class:`qsa.domain.Subset`)
    :type subset: enumerate
    :return: interactive frequencies (hertz) generated from fundamental frequencies
    :rtype: numpy.ndarray

    .. highlight:: python
    .. code-block:: python

        >>> qsa.domain.order(numpy.array([2, 7]), True, qsa.domain.Subset.QUADRATIC)
        array([-14.,  -9.,  -5.,  -4.,   4.,   5.,   9.,  14.])
    """
    mixing = numpy.array([])
    for i in range(0, len(frequencies)):
        fi = frequencies[i]
        if Subset.LINEAR.value & subset.value:
            mixing = numpy.append(mixing, fi)
        if Subset.DIAGONAL.value & subset.value:
            mixing = numpy.append(mixing, 2 * fi)
        for j in range(i + 1, len(frequencies)):
            fj = frequencies[j]
            if Subset.PLUS.value & subset.value:
                mixing = numpy.append(mixing, fi + fj)
            if Subset.MINUS.value & subset.value:
                mixing = numpy.append(mixing, abs(fi - fj))
    result = mixing
    if symmetric:
        negative = numpy.negative(result)
        result = numpy.concatenate([negative, result])
    return numpy.sort(result)
