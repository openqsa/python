"""
This module represents an RTXI experiment as an array of measurements (see also :py:class:`qsa.measurement.Measurement`).
"""

import json
import numpy

import qsa.measurement


class Experiment:
    """
    Representation of an RTXI experiment as an array of measurements.
    Each measurement corresponds to the multisinusoidal part of a single trace.
    """

    def __init__(self, filename):
        """
        Constructor.

        :param filename: Filename of the JSON file generated by the RTXI program.
        :type filename: str
        """
        with open(filename, 'r') as file:
            text = file.read()
        j = json.loads(text)
        self.__version = j['version']
        self.__dt = j['dt']
        self.__duration = j['duration']
        self.__frequencies = numpy.array(j['frequencies'])
        self.__amplitudes = numpy.array(j['amplitudes'])
        self.__phases = numpy.array(j['phases'])
        self.__rest_level = j['rest_level']
        self.__step_level = j['step_level']
        self.__step_delay = j['step_delay']
        self.__drop_delay = j['drop_delay']
        self.__trace_count = j['trace_count']
        self.__trace_pause = j['trace_pause']
        self.__trace_alternance = j['trace_alternance']
        self.__traces = []
        for i in range(0, len(j['traces'])):
            def trace(): return None

            trace.step = lambda: None
            trace.step.time = numpy.array(j['traces'][i]['step']['time'])
            trace.step.stimulation = numpy.array(
                j['traces'][i]['step']['stimulation'])
            trace.step.response = numpy.array(
                j['traces'][i]['step']['response'])
            trace.multisine = lambda: None
            trace.multisine.time = numpy.array(
                j['traces'][i]['multisine']['time'])
            trace.multisine.stimulation = numpy.array(
                j['traces'][i]['multisine']['stimulation'])
            trace.multisine.response = numpy.array(
                j['traces'][i]['multisine']['response'])
            trace.drop = lambda: None
            trace.drop.time = numpy.array(j['traces'][i]['drop']['time'])
            trace.drop.stimulation = numpy.array(
                j['traces'][i]['drop']['stimulation'])
            trace.drop.response = numpy.array(
                j['traces'][i]['drop']['response'])
            self.__traces.append(trace)

    @property
    def version(self):
        """
        Get version.

        :return: version
        :rtype: str
        """
        return self.__version

    def count(self):
        """
        Count measurements.

        :return: number of measurements
        :rtype: int
        """
        return len(self.__traces)

    def get_measurement(self, index):
        """
        Get measurement.

        :param index: index of measurement in array of measurements
        :type index: int
        :return: measurement at the specified index
        :rtype: qsa.measurement.Measurement
        :raises IndexError: if index is out of range
        """
        if index < 0 or index >= len(self.__traces):
            raise IndexError('Invalid measurement index (%d)' % index)
        return qsa.measurement.Measurement(
            self.__dt,
            self.__duration,
            self.__frequencies,
            self.__traces[index].multisine.time,
            self.__traces[index].multisine.stimulation,
            self.__traces[index].multisine.response)

    def average(self, indices):
        """
        Compute averaging of several measurements.

        :param indices: array of measurement indices to take into account in averaging
        :type indices: array
        :return: averaged measurement
        :rtype: qsa.measurement.Measurement
        :raises NotImplementedError: if trace_alternance < 0
        :raises IndexError: if index is out of range
        """
        if self.__trace_alternance < 0:
            raise NotImplementedError(
                'Averaging with trace alternance is not yet implemented')
        t = 0
        x = 0
        y = 0
        n = len(indices)
        for i in range(0, n):
            index = indices[i]
            if index < 0 or index >= len(self.__traces):
                raise IndexError('Invalid measurement index (%d)' % index)
            m = self.get_measurement(index)
            t = t + m.t / n
            x = x + m.x / n
            y = y + m.y / n
        return qsa.measurement.Measurement(
            self.__dt,
            self.__duration,
            self.__frequencies,
            t,
            x,
            y)
