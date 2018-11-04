"""
This modules provides graphical tools to display analysis results.
All functions are based on the matplotlib plotting library.
"""

import numpy
import matplotlib.pyplot as plt

import qsa.domain


def ion():
    """
    Turn the interactive mode on.

    :return: result of :py:func:`matplotlib.pyplot.ion`
    """
    plt.ion()


def figure():
    """
    Create a new figure.

    :return: result of :py:func:`matplotlib.pyplot.figure`
    """
    plt.figure()


def subplot(nrows, ncols, index):
    """
    Add a subplot to the current figure.

    :param nrows: number of rows
    :type nrows: int
    :param ncols: number of columns
    :type ncols: int
    :param index: index of row and column ; starts at 1 in the upper left corner and increases to the right
    :type index: int
    :return: result of :py:func:`matplotlib.pyplot.subplot`
    """
    plt.subplot(nrows, ncols, index)


def plot_stimulation(a):
    """
    Plot the stimulation in the time and frequency domain.
    A new figure with subplots is automatically created.

    :param a: QSA analysis
    :type a: qsa.analysis.Analysis
    :return: None
    """
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(a.t, a.x)
    plt.title('Stimulation')
    plt.xlabel('Time (s)')
    plt.ylabel('Input')
    plt.tight_layout()
    plt.subplot(2, 1, 2)
    plt.plot(numpy.abs(a.x_sp))
    plt.title('Stimulation (FFT)')
    plt.xlabel('Frequency (Hertz)')
    plt.ylabel('Input')
    plt.tight_layout()


def plot_response(a):
    """
        Plot the response in the time and frequency domain.
        A new figure with subplots is automatically created.

        :param a: QSA analysis
        :type a: qsa.analysis.Analysis
        :return: None
    """
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(a.t, a.y)
    plt.title('Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.tight_layout()
    plt.subplot(2, 1, 2)
    plt.plot(numpy.abs(a.y_sp))
    plt.title('Response (FFT)')
    plt.xlabel('Frequency (Hertz)')
    plt.ylabel('Output')
    plt.tight_layout()


def plot_comparison(a, legend=True, new_figure=True):
    """
    Plot the comparison between response, linear and linear + quadratic analyses in the time domain.
    The graphic is embedded in the current figure, or a new figure can be created.

    :param a: QSA analysis
    :type a: qsa.analysis.Analysis
    :param legend: True (default) if legend should be displayed
    :type legend: bool
    :param new_figure: True (default) if a new figure should be created
    :type new_figure: bool
    :return: None
    """
    if new_figure:
        plt.figure()
    plt.plot(a.t, a.y, c='b')
    plt.plot(a.t, a.y1, c='g')
    plt.plot(a.t, a.y12, c='r')
    plt.title('Comparison', fontsize=16)
    plt.xlabel('Frequency (Hertz)')
    plt.ylabel('Responses')
    if legend:
        plt.legend(['Y', 'Y1', 'Y12'])
    plt.tight_layout()


def plot_pure(a, new_figure=True):
    """
    Plot the pure quadratic analysis in the time domain.
    The pure quadratic analysis has no dc and no linear component.
    The graphic is embedded in the current figure, or a new figure can be created.

    :param a: QSA analysis
    :type a: qsa.analysis.Analysis
    :param new_figure: True (default) if a new figure should be created
    :type new_figure: bool
    :return: None
    """
    if new_figure:
        plt.figure()
    plt.plot(a.t, a.y2 - a.y_dc)
    plt.title('Pure quadratic')
    plt.xlabel('Frequency (Hertz)')
    plt.ylabel('Response Y2')
    plt.tight_layout()


def plot_transfer(a, new_figure=True):
    """
    Plot the linear transfer function in the frequency domain.
    The function represented is :math:`\\left|\\frac{Y(\\omega_k)}{X(\\omega_k)}\\right|`
    where :math:`\\omega_k` describe the finite set of positive linear frequencies.
    The graphic is embedded in the current figure, or a new figure can be created.

    :param a: QSA analysis
    :type a: qsa.analysis.Analysis
    :param new_figure: True (default) if a new figure should be created
    :type new_figure: bool
    :return: None
    """
    if new_figure:
        plt.figure()
    plt.stem(a.frequencies1, numpy.abs(a.linear), linefmt='gray')
    plt.plot(a.frequencies1, numpy.abs(a.linear), c='b')
    plt.title('Linear transfer', fontsize=16)
    plt.xlabel('Frequency (Hertz)')
    plt.ylabel('Magnitude')
    plt.tight_layout()


def plot_flat(a, subset=qsa.domain.Subset.ALL, legend=True, new_figure=True):
    """
    Plot the response in the frequency domain.
    The function represented is :math:`\\left|Y(\\Omega_k)\\right|`
    where :math:`\\Omega_k` is a subset of positive interactive frequencies.
    The graphic is embedded in the current figure, or a new figure can be created.

    :param a: QSA analysis
    :type a: qsa.analysis.Analysis
    :param subset: subset of interactive frequencies
    :type subset: qsa.domain.Subset
    :param legend: True (default) if legend should be displayed
    :type legend: bool
    :param new_figure: True (default) if a new figure should be created
    :type new_figure: bool
    :return: None
    """
    if new_figure:
        plt.figure()
    labels = []
    if qsa.domain.Subset.LINEAR.value & subset.value:
        plt.plot(a.frequencies1, numpy.abs(a.y_sp1), c='k')
        labels.append('Y1')
    if qsa.domain.Subset.DIAGONAL.value & subset.value:
        plt.scatter(a.frequencies2d, numpy.abs(a.y_sp2d), c='g')
        labels.append('Y2 diag')
    if qsa.domain.Subset.PLUS.value & subset.value:
        plt.scatter(a.frequencies2p, numpy.abs(a.y_sp2p), c='r')
        labels.append('Y2 plus')
    if qsa.domain.Subset.MINUS.value & subset.value:
        plt.scatter(a.frequencies2m, numpy.abs(a.y_sp2m), c='b')
        labels.append('Y2 minus')
    plt.title('Flat response', fontsize=16)
    plt.xlabel('Frequency (Hertz)')
    plt.ylabel('Magnitude')
    if legend:
        plt.legend(labels)
    plt.tight_layout()


def plot_qmatrix(a, new_figure=True):
    """
    Plot the quadratic transfer function in the frequency domain.
    The function represented is :math:`\\left|Q_{i,j}\\right|`
    where :math:`(Q_{i,j})` is the QSA matrix.
    The graphic is embedded in the current figure, or a new figure can be created.

    :param a: QSA analysis
    :type a: qsa.analysis.Analysis
    :param new_figure: True (default) if a new figure should be created
    :type new_figure: bool
    :return: None
    """
    if new_figure:
        plt.figure()
    plt.matshow(numpy.abs(a.quadratic), cmap='jet', fignum=False)
    plt.colorbar()
    plt.title('QSA matrix', fontsize=16)
    gca = plt.gca()
    gca.set_xticklabels([])
    gca.set_yticklabels([])
    plt.tight_layout()


def plot_qeigen(a, new_figure=True):
    """
    Plot the eigenvalues of the QSA matrix.
    These are sorted by magnitude. Note that eigenvalues are real since the QSA matrix is Hermitian.

    :param a: QSA analysis
    :type a: qsa.analysis.Analysis
    :param new_figure: True (default) if a new figure should be created
    :type new_figure: bool
    :return: None
    """
    if new_figure:
        plt.figure()
    eigen = sorted(a.eigen, key=numpy.abs, reverse=True)
    plt.bar(numpy.arange(len(eigen)), eigen)
    plt.title('QSA eigenvalues')
    plt.tight_layout()
