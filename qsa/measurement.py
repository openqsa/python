"""
This module represents a single trace measurement in the stimulation-response paradigm.
A measurement is applicable to both physical and virtual experiments.
"""

import numpy


class Measurement:
    """
    Implementation of a single trace measurement.
    """

    def __init__(self, dt, duration, frequencies, t, x, y):
        """
        Constructor.

        :param dt: sampling time (seconds)
        :type dt: float
        :param duration: experiment duration (seconds) such that :math:`1 / duration` represents the minimal frequency
        :type duration: float
        :param frequencies: array of fundamental frequencies (hertz)
        :type frequencies: numpy.ndarray
        :param t: time vector (seconds)
        :type t: numpy.ndarray
        :param x: stimulation vector
        :type x: numpy.ndarray
        :param y: response vector
        :type y: numpy.ndarray
        :raises ValueError: if t,x,y have wrong size
        """
        if len(t) != round(duration / dt):
            raise ValueError("Invalid length (%f) for t vector" % len(t))
        if len(x) != len(t):
            raise ValueError("Invalid length (%f) for x vector" % len(x))
        if len(y) != len(t):
            raise ValueError("Invalid length (%f) for y vector" % len(y))
        self.__dt = dt
        self.__duration = duration
        self.__frequencies = numpy.copy(frequencies)
        self.__t = numpy.copy(t)
        self.__x = numpy.copy(x)
        self.__y = numpy.copy(y)

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
