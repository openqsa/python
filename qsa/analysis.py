"""
This module computes linear and quadratic analyses of measurements.
"""

import numpy

import qsa.domain
import qsa.measurement


class Analysis:
    """
    Computation of linear and quadratic analysis of a measurement.
    """

    def __init__(self, measurement):
        """
        Constructor.

        :param measurement: the measurement to be analyzed
        :type measurement: qsa.measurement.Measurement
        """
        self.__dt = measurement.dt
        self.__duration = measurement.duration
        self.__frequencies = numpy.copy(measurement.frequencies)
        self.__t = numpy.copy(measurement.t)
        self.__x = numpy.copy(measurement.x)
        self.__y = numpy.copy(measurement.y)
        #
        self.__x_sp = spectralize(self.__x)
        self.__y_sp = spectralize(self.__y)
        #
        self.__x_dc = numpy.real(self.__x_sp[0])
        self.__y_dc = numpy.real(self.__y_sp[0])
        #
        self.__frequencies1 = qsa.domain.order(
            self.__frequencies,
            symmetric=False,
            subset=qsa.domain.Subset.LINEAR)
        self.__ficks1 = qsa.domain.hertz2ficks(
            self.__dt, self.__duration, self.__frequencies1)
        self.__x_sp1 = self.__x_sp[self.__ficks1]
        self.__x1 = self.__approximate(self.__x_sp, qsa.domain.Subset.LINEAR)
        self.__y_sp1 = self.__y_sp[self.__ficks1]
        self.__y1 = self.__approximate(self.__y_sp, qsa.domain.Subset.LINEAR)
        #
        self.__frequencies2 = qsa.domain.order(
            self.__frequencies,
            symmetric=False,
            subset=qsa.domain.Subset.QUADRATIC)
        self.__ficks2 = qsa.domain.hertz2ficks(
            self.__dt, self.__duration, self.__frequencies2)
        self.__y_sp2 = self.__y_sp[self.__ficks2]
        self.__y2 = self.__approximate(self.__y_sp, qsa.domain.Subset.QUADRATIC)
        #
        self.__frequencies12 = qsa.domain.order(
            self.__frequencies,
            symmetric=False,
            subset=qsa.domain.Subset.ALL)
        self.__ficks12 = qsa.domain.hertz2ficks(
            self.__dt, self.__duration, self.__frequencies12)
        self.__y_sp12 = self.__y_sp[self.__ficks12]
        self.__y12 = self.__approximate(self.__y_sp, qsa.domain.Subset.ALL)
        #
        self.__frequencies2d = qsa.domain.order(
            self.__frequencies,
            symmetric=False,
            subset=qsa.domain.Subset.DIAGONAL)
        self.__ficks2d = qsa.domain.hertz2ficks(
            self.__dt, self.__duration, self.__frequencies2d)
        self.__y_sp2d = self.__y_sp[self.ficks2d]
        self.__frequencies2p = qsa.domain.order(
            self.__frequencies,
            symmetric=False,
            subset=qsa.domain.Subset.PLUS)
        self.__ficks2p = qsa.domain.hertz2ficks(
            self.__dt, self.__duration, self.__frequencies2p)
        self.__y_sp2p = self.__y_sp[self.__ficks2p]
        self.__frequencies2m = qsa.domain.order(
            self.__frequencies,
            symmetric=False,
            subset=qsa.domain.Subset.MINUS)
        self.__ficks2m = qsa.domain.hertz2ficks(
            self.__dt, self.__duration, self.__frequencies2m)
        self.__y_sp2m = self.__y_sp[self.__ficks2m]
        #
        self.__linear = self.__y_sp1 / self.__x_sp1
        self.__quadratic = self.__qmatrix()
        self.__eigen = numpy.linalg.eigvalsh(self.__quadratic)

    @property
    def dt(self):
        """
        Get sampling time.

        :return: dt (seconds)
        :rtype: float
        """
        return self.__dt

    @property
    def duration(self):
        """
        Get experiment duration.

        :return: duration (seconds)
        :rtype: float
        """
        return self.__duration

    @property
    def frequencies(self):
        """
        Get fundamental frequencies.

        :return: copy of the array of fundamental frequencies (hertz)
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__frequencies)

    @property
    def t(self):
        """
        Get time.

        :return: copy of the time vector (seconds)
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__t)

    @property
    def x(self):
        """
        Get stimulation.

        :return: copy of the stimulation vector
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__x)

    @property
    def y(self):
        """
        Get response.

        :return: copy of the response vector
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y)

    @property
    def x_sp(self):
        """
        Get stimulation spectrum.

        :return: copy of the stimulation spectrum vector
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__x_sp)

    @property
    def y_sp(self):
        """
        Get response spectrum.

        :return: copy of the response spectrum vector
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y_sp)

    @property
    def x_dc(self):
        """
        Get stimulation DC.

        :return: stimulation DC
        :rtype: float
        """
        return self.__x_dc

    @property
    def y_dc(self):
        """
        Get response DC.

        :return: response DC
        :rtype: float
        """
        return self.__y_dc

    @property
    def frequencies1(self):
        """
        Get positive linear frequencies.

        :return: copy of the array of positive linear frequencies (hertz)
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__frequencies1)

    @property
    def ficks1(self):
        """
        Get FFT indices of frequencies1.

        :return: copy of the array of FFT indices of frequencies1
        :rtype: numpy.ndarray
        """
        return self.copy(self.__ficks1)

    @property
    def x_sp1(self):
        """
        Get stimulation spectrum over ficks1.

        :return: copy of the stimulation spectrum over ficks1
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__x_sp1)

    @property
    def x1(self):
        """
        Get linear approximation of stimulation

        :return: copy of the linear approximation of stimulation
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__x1)

    @property
    def y_sp1(self):
        """
        Get response spectrum over ficks1.

        :return: copy of the response spectrum over ficks1
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y_sp1)

    @property
    def y1(self):
        """
        Get dc + linear approximation of response

        :return: copy of the dc + linear approximation of response
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y1)

    @property
    def frequencies2(self):
        """
        Get positive quadratic frequencies.

        :return: copy of the array of positive quadratic frequencies (hertz)
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__frequencies2)

    @property
    def ficks2(self):
        """
        Get FFT indices of frequencies2.

        :return: copy of the array of FFT indices of frequencies2
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__ficks2)

    @property
    def y_sp2(self):
        """
        Get response spectrum over ficks2.

        :return: copy of the response spectrum over ficks2
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y_sp2)

    @property
    def y2(self):
        """
        Get dc + quadratic approximation of response

        :return: copy of the dc + quadratic approximation of response
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y2)

    @property
    def frequencies12(self):
        """
        Get positive linear + quadratic frequencies.

        :return: copy of the array of positive linear + quadratic frequencies (hertz)
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__frequencies12)

    @property
    def ficks12(self):
        """
        Get FFT indices of frequencies12.

        :return: copy of the array of FFT indices of frequencies12
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__ficks12)

    @property
    def y_sp12(self):
        """
        Get response spectrum over ficks12.

        :return: copy of the response spectrum over ficks12
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y_sp12)

    @property
    def y12(self):
        """
        Get dc + linear + quadratic approximation of response

        :return: copy of the dc + linear + quadratic approximation of response
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y12)

    @property
    def frequencies2d(self):
        """
        Get positive diagonal frequencies.

        :return: copy of the array of positive diagonal frequencies (hertz)
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__frequencies2d)

    @property
    def ficks2d(self):
        """
        Get FFT indices of frequencies2d.

        :return: copy of the array of FFT indices of frequencies2d
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__ficks2d)

    @property
    def y_sp2d(self):
        """
        Get response spectrum over ficks2d.

        :return: copy of the response spectrum over ficks2d
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y_sp2d)

    @property
    def frequencies2p(self):
        """
        Get positive plus frequencies.

        :return: copy of the array of positive plus frequencies (hertz)
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__frequencies2p)

    @property
    def ficks2p(self):
        """
        Get FFT indices of frequencies2p.

        :return: copy of the array of FFT indices of frequencies2p
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__ficks2p)

    @property
    def y_sp2p(self):
        """
        Get response spectrum over ficks2p.

        :return: copy of the response spectrum over ficks2p
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y_sp2p)

    @property
    def frequencies2m(self):
        """
        Get positive minus frequencies.

        :return: copy of the array of positive minus frequencies (hertz)
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__frequencies2m)

    @property
    def ficks2m(self):
        """
        Get FFT indices of frequencies2m.

        :return: copy of the array of FFT indices of frequencies2m
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__ficks2m)

    @property
    def y_sp2m(self):
        """
        Get response spectrum over ficks2m.

        :return: copy of the response spectrum over ficks2m
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__y_sp2m)

    @property
    def linear(self):
        """
        Get linear transfer function over ficks1.

        :return: copy of the transfer function over ficks1
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__linear)

    @property
    def quadratic(self):
        """
        Get quadratic transfer function, which is the QSA matrix.

        :return: copy of the  quadratic transfer function
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__quadratic)

    @property
    def eigen(self):
        """
        Get eigenvalues of the QSA matrix.

        :return: copy of the eigenvalues
        :rtype: numpy.ndarray
        """
        return numpy.copy(self.__eigen)

    def __bcoefficient(self, f, i, j):
        if f[i] == - f[j]:
            return 0
        k = 1 if i == j else 0.5
        fi = qsa.domain.hertz2fick(self.dt, self.duration, f[i])
        fj = qsa.domain.hertz2fick(self.dt, self.duration, f[j])
        fij = qsa.domain.hertz2fick(self.dt, self.duration, f[i] + f[j])
        rij = self.y_sp[fij]
        si = self.x_sp[fi]
        sj = self.x_sp[fj]
        return k * rij / (si * sj)

    def __bmatrix(self):
        n = len(self.frequencies)
        b = numpy.zeros([2 * n, 2 * n], dtype=complex)
        f = qsa.domain.order(
            self.frequencies,
            symmetric=True,
            subset=qsa.domain.Subset.LINEAR)
        for i in range(0, 2 * n):
            for j in range(0, 2 * n):
                b[i, j] = self.__bcoefficient(f, i, j)
        return b

    def __qmatrix(self):
        return numpy.flipud(self.__bmatrix())

    def __approximate(self, z_sp, subset):
        frequencies_generated = qsa.domain.order(
            self.frequencies,
            symmetric=True,
            subset=subset)
        ficks = qsa.domain.hertz2ficks(
            self.dt,
            self.duration,
            frequencies_generated)
        za_sp = numpy.zeros(numpy.size(z_sp), dtype='complex')
        za_sp[0] = z_sp[0]
        za_sp[ficks] = z_sp[ficks]
        return unspectralize(za_sp)


def spectralize(z):
    """
    Compute discrete Fourier transform of a real signal.
    The FFT is normalized by the length of the signal. In particular,
    :math:`cos(t)` is transformed to a sum of two complex exponentials with half
    amplitude

    .. math::

        cos(t) = \\frac{1}{2} e^{i t} + \\frac{1}{2} e^{- i t}

    :param z: signal as a time series of real values
    :type z: numpy.ndarray
    :return: array of Fourier coefficients (complex numbers)
    :rtype: numpy.ndarray

    .. highlight:: python
    .. code-block:: python

        >>> t = numpy.linspace(0, 10, 10000)
        >>> z = numpy.cos(2 * numpy.pi * t)
        >>> z_sp = qsa.analysis.spectralize(z)
        >>> numpy.max(numpy.abs(z_sp))
        0.5000241787423103
    """
    return numpy.fft.fft(z) / len(z)


def unspectralize(z_sp):
    """
    Compute inverse of a Fourier transform obtained with :py:func:`spectralize`.
    By definition, unspectralize(spectralize(z)) equals z.

    :param z_sp: a Fourier transform obtained with :py:func:`spectralize`
    :type z_sp: numpy.ndarray
    :return: a real signal for which Fourier transform is z_sp
    :rtype: numpy.ndarray

    .. highlight:: python
    .. code-block:: python

        >>> t = numpy.linspace(0, 10, 10000)
        >>> z = numpy.cos(2 * numpy.pi * t)
        >>> z_sp = qsa.analysis.spectralize(z)
        >>> z_sp_inv = qsa.analysis.unspectralize(z_sp)
        >>> numpy.max(z - z_sp_inv)
        7.771561172376096e-16
    """
    z_sp_pure = numpy.copy(z_sp)
    z_sp_pure[0] = 0
    return numpy.real(numpy.fft.ifft(z_sp_pure) * len(z_sp) + z_sp[0])
