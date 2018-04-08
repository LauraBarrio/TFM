import numpy as np


def calculate_bins(color_list, bw):
    """
    Calculates the bin interval for a cluster using the bp-rp member values

    :param color_list: bp-rp values
    :param bw: interval width
    :return: bin values
    """
    start = min(color_list)
    end = max(color_list)+0.1
    bw_i = (end - start) / bw

    point_bins = np.arange(start, end + bw_i, bw_i)

    bins = []

    for i in range(bw):
        bins.append([round(point_bins[i], 1), round(point_bins[i + 1], 1)])

    return bins
