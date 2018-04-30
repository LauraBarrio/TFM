"""
@author: M. Palmer

Computes the ML estimates of the parameters of the model being the stars belonging to a cluster which is spherical at
some distance, with the absolute magnitude following a isochrone and moving trough the space with some velocity.

Input: a file with a list of cluster members with the following parameters:
magnitud G        = mg (mag)
galatic latitude  = l (radians)
galatic longitude = b (radians)
parallax          = pi (muas)
parallax error    = epsPi (muas)
mualpha           = muAlpha (muas)
mualpha error     = epsMuAlpha (muas)
mudelta           = muDelta (muas)
mudelta error     = epsMuDelta (muas)
color             = bp-rp (mag) (bt-vt can be used)

Output:
Parameters, distance, position, speed, size isochrone, with their errors
labels =  ['R','sS1','sS2','sS3','sS4','x1','x2','x3','x4','x5','sM1','sM2','sM3','sM4','muAlphaMean','muDeltaMean','sigmaMuAlpha','sigmaMuDelta']

"""

import random
import time
from functools import partial
from math import pow, pi, sqrt, log10, exp, sin, cos
from multiprocessing import Pool

import numdifftools as nd
import numpy as np
import pandas as pd
from numpy import matrix
from scipy import interpolate
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
from scipy.special import erfc

from Functions3 import getParameters, getSigmaSSigmaMforC, getWeight, AgCluster, getSigmaSSigmaMforD, getCminCmaxforD, \
    epsMuAlpha, epsMuDelta, Fmag, Fvel, jacobian, Fspace, epsPi, checkParamsAreValid, getSpline, \
    transformCoordinates, logD, findWeightings, calculateMeanL, calculateMeanB

from bins import calculate_bins

random.seed(1)


def calculateErrors(starlist, c, res, bins):
    """
    Finds the derivative of the likelihood function around the maximum computing a Hessian matrix
    (Please refer to M. Palmer et al (2014)

    :param starlist: star list
    :param c: star dictionary with number of stars, Gaia mag limit, lMean, bMean, w
    :param res: MLE results
    :param bins: bin list
    :return: covariance matrix
    """
    print "\n\nCalculating errors...\n\n"

    def callML(x):
        p = getParameters(x)
        return LikelihoodFunction(starlist, p, c, bins)

    def makearray(value, nrows, ncols):
        return [[value] * ncols for _ in range(nrows)]

    cov = makearray(None, 16, 16)
    error = []
    Hfun = nd.Hessian(callML)
    tempMatrix = matrix(Hfun(res))

    print tempMatrix
    covarianceMatrix = np.array(tempMatrix.I).tolist()
    print "\nCovariance Matrix:\n", covarianceMatrix

    for i in xrange(0, len(res), 1):
        try:
            error.append(sqrt(-covarianceMatrix[i][i]))
        except:
            error.append("Error")
            print "Hessian Error"
    print "Errors are:", error

    for i in xrange(0, len(res), 1):
        for j in xrange(0, len(res), 1):
            try:
                cov[i][j] = covarianceMatrix[i][j] / (error[i] * error[j])
            except:
                print i, j
                cov[i][j] = "Error"
    print "correlations are", cov


def calculatesNormalizationCoeffPool(colours, parameters, constants, spline, bins):
    """
    Calculates the normalisation coefficient

    :param colours:
    :param parameters: cluster parameters
    :param constants: cluster constants
    :param spline: spline
    :param bins: bin list
    :return: normalisation coefficient for each bins item
    """

    minColour = colours[0]
    maxColour = colours[1]

    sigmaS, sigmaM = getSigmaSSigmaMforC(parameters, minColour, maxColour, bins)
    weight = getWeight(constants, minColour, maxColour, bins)

    def Ir(r0):
        def Ib(b0):
            def Il(l0): return exp(
                (
                    -0.5 * (parameters['R^2'] + r0 ** 2 - 2 * r0 * parameters['R'] * cos(b0) * cos(l0)) / (
                        pow(sigmaS, 2))))

            integralL = quad(Il, -0.04, 0.04)
            return integralL[0] * cos(b0)

        def Ibprp(bprp):
            Mh = interpolate.splev(bprp, spline)
            m0 = Mh + (5 * log10(r0)) - 5 + AgCluster
            p = sigmaM * erfc(-((constants['mlim'] - m0) / (sqrt(2) * sigmaM)))
            return p

        integralB = quad(Ib, -0.04, 0.04)
        integralBPRP = quad(Ibprp, minColour, maxColour)

        return integralB[0] * r0 ** 2 * integralBPRP[0]

    integralR = quad(Ir, 10, parameters['R'] + 100, epsabs=0)

    return 0.5 * (2 * pi) ** 4 * parameters['sigmaMuAlpha'] * parameters['sigmaMuDelta'] * integralR[0] * weight


def calculatesUnnormaLikelihoodPool(star, parameters, constants, spline, CsinEps, bins):
    """
    Calculates the unnormalized likelihood for each star

    :param parameters: cluster parameters
    :param constants: cluster constants
    :param star: star
    :param spline: spline
    :param CsinEps: sum of the normalisation coefficients
    :param bins: bin list
    :return: probability of the unnormalized likelihood
    """

    star_t = transformCoordinates(star, constants)
    sigmaS, sigmaM = getSigmaSSigmaMforD(parameters, star_t, bins)
    thisCmin, thisCmax = getCminCmaxforD(star_t, bins)
    weight = getWeight(constants, thisCmin, thisCmax, bins)

    def Imu(muAlpha0, muDelta0): return Fvel(muAlpha0, muDelta0, parameters['muAlphaMean'], parameters['sigmaMuAlpha'],
                                             parameters['muDeltaMean'],
                                             parameters['sigmaMuDelta']) * epsMuAlpha(star_t, muAlpha0) * epsMuDelta(
        star_t,
        muDelta0)

    integralMu = dblquad(Imu, parameters['muDeltaMean'] - 0.025, parameters['muDeltaMean'] + 0.025,
                         lambda x: parameters['muAlphaMean'] - 0.025, lambda x: parameters['muAlphaMean'] + 0.025,
                         epsabs=0)

    def Ir(r0): return jacobian(star_t, r0) * Fspace(star_t, r0, parameters, sigmaS) * Fmag(star_t, r0, spline,
                                                                                            sigmaM) * epsPi(
        star_t, r0) * integralMu[0]

    integralR = quad(Ir, 10, parameters['R'] + 50)

    return logD(weight * integralR[0] / (CsinEps * star_t['epsPi'] * star_t['epsMuAlpha'] * star_t['epsMuDelta']))


def LikelihoodFunction(starlist, parameters, constants, bins):
    """
    Calculates the value of the likelihood function for every star

    :param starlist:
    :param parameters: cluster parameters
    :param constants: cluster constants
    :param bins: bin list
    :return: final sum of the MLE
    """

    if not checkParamsAreValid(parameters):
        print "The Sum: -10E307 \n"
        return -10E307
    else:

        spline = getSpline(bins, parameters)

        # t0 used for the total time
        t0 = time.time()
        # Number maximum of processes is 4 (bins is a list of 4 items)
        pool = Pool(processes=4)

        result_normalisation_coef = pool.map(
            partial(calculatesNormalizationCoeffPool, parameters=parameters, constants=constants, spline=spline,
                    bins=bins), bins)

        # Closing the pool
        pool.close()
        # Joining the results
        pool.join()

        # Sum of the normalisation coefficient that we use as denominator for the likelihood function
        CsinEps = sum(result_normalisation_coef)

        # This number of processes can be changed depending on the number of cores. Theoretically, the maximum number of
        # processes can be as high as the number of stars (this was the first implementation) but that can compromises
        # the capacity of the CPU and it is not very optimal
        pool = Pool(processes=4)
        result_likelihood = pool.map(
            partial(calculatesUnnormaLikelihoodPool, parameters=parameters, constants=constants, spline=spline,
                    CsinEps=CsinEps, bins=bins), starlist)

        # Closing the pool
        pool.close()
        # Joining the results
        pool.join()

        finalSum = np.sum(result_likelihood)

        # Print used for experiment with the number of processes, it can be removed
        print "Total time: " + str(time.time() - t0)

        print "The Sum:", finalSum, "\n"
        return finalSum


def MLE(starlist, constants, guess, bins):
    """
    MLE wrapper

    :param starlist: star list
    :param constants: dictionary with the constants
    :param guess: initial guesses
    :param bins: bin list
    :return:
    """

    t0 = time.time()
    print "\nMean coordinates are", constants['lmean'], constants['bmean']

    def f(x):
        print x
        parameters = getParameters(x)
        return -LikelihoodFunction(starlist, parameters, constants, bins)

    res = minimize(f, x0=guess, method="Powell")
    print "NM: ", res.message, "\n", res.x, "\n", "Total time:", time.time() - t0
    return res.x


"""

Main program to calculate ML. Catalogue should be located inside the same directory

"""
if __name__ == '__main__':

    # Reads a csv file containing the parameters
    fileName = 'Pleiades'
    starlist = pd.read_csv(fileName, names=['mg', 'l', 'b', 'pi', 'epsPi',
                                            'muAlpha', 'epsMuAlpha', 'muDelta',
                                            'epsMuDelta', 'bp-rp']).T.to_dict().values()

    # Coordinate transformation to 3 dimensional space x, y and z
    for star in starlist:
        star['sinl'] = sin(star['l'])
        star['cosl'] = cos(star['l'])
        star['sinb'] = sin(star['b'])
        star['cosb'] = cos(star['b'])
        star['x'] = (1.0 / star['pi']) * star['cosb'] * star['cosl']
        star['y'] = (1.0 / star['pi']) * star['cosb'] * star['sinl']
        star['z'] = (1.0 / star['pi']) * star['sinb']

    # In case of having radial velocities
    # for i in range(len(starlist)):
    #    starlist[i].pop('vr', None)
    #    starlist[i].pop('epsVr-V', None)

    distlist, maxlist, minlist, llist, blist, xlist, ylist, \
    zlist, bp_rp_list, mualpha_list, mudelta_list = [], [], [], [], [], [], [], [], [], [], []

    # Creates list of values for each parameter
    for star in starlist:
        if star['pi'] > 0:
            distlist.append(1 / star['pi'])
            llist.append(star['l'])
            blist.append(star['b'])
            maxlist.append((1 / star['pi']) - (1 / (star['pi'] - star["epsPi"])))
            minlist.append(-((1 / star['pi']) - (1 / (star['pi'] + star["epsPi"]))))
            xlist.append(star['x'])
            ylist.append(star['y'])
            zlist.append(star['z'])
        bp_rp_list.append(star['bp-rp'])
        mualpha_list.append(star['muAlpha'])
        mudelta_list.append(star['muDelta'])

    # x, y and z means
    meanx = np.mean(xlist)
    meany = np.mean(ylist)
    meanz = np.mean(zlist)
    print "mean cluster coordinates for original catalogue:", meanx, meany, meanz

    # Dynamic initial guesses
    # Distance
    r_i = np.mean(distlist)
    # Mualpha and mudelta
    mualpha_m_i = np.mean(mualpha_list)
    mudelta_m_i = np.mean(mudelta_list)

    if fileName == 'Hyades':
        # Initial Guess for Hyades
        initial_guess = [r_i, 3.41991717, 5.50180935, 2.29546342, 2.54500714, 0.76959647, 3.98301261,
                         5.24713807, 7.05108992, 9.25966225, 0.39008765, 0.32244944, 0.63084539, 0.34296494,
                         mualpha_m_i, mudelta_m_i, 10,
                         10]
    if fileName == 'Pleiades':
        # Initial Guess for Pleiades
        initial_guess = [r_i, 3.35150575e+00, 4.86142031e+00, 1.03207132e+01, 1.31350468e+01,
                         -2.39384409e+00, 1.96855446e-01, 3.04079043e+00, 4.22140557e+00, 5.37379397e+00,
                         1.55576559e+00, 4.49615136e-01, 2.19895913e-01, 1.67554634e-01, mualpha_m_i, mudelta_m_i, 0.1,
                         0.1]
    if fileName == "Alpha persei":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.10, 0.30, 0.70, 1.10, 1.50, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0018400, 0.0015100]
    if fileName == "Blanco1":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.10, 0.20, 0.50, 0.80, 1.10, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0008200, 0.0008200]
    if fileName == "Col140":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.30, 0.20, 0.70, 1.20, 1.70, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0006100, 0.0007300]
    if fileName == "Coma berenices":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, 0.10, 0.60, 1.10, 1.60, 2.10, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0013700, 0.0011600, ]
    if fileName == "IC2391":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.20, 0.10, 0.40, 0.70, 1.00, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0027600, 0.0027290]
    if fileName == "IC2602":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.30, 0.30, 0.90, 1.50, 2.10, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0022600, 0.0030900]
    if fileName == "IC4665":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, 0.10, 0.30, 0.70, 1.10, 1.50, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0003200, 0.0002900]
    if fileName == "NGC2232":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.20, 0.10, 0.40, 0.70, 1.00, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0004400, 0.0005700]
    if fileName == "NGC2422":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.20, 0.10, 0.40, 0.70, 1.00, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0004700, 0.0003200]
    if fileName == "NGC2451":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.15, 0.26, 0.67, 1.08, 1.49, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0012100, 0.0011100]
    if fileName == "NGC2516":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.10, 0.15, 0.40, 0.65, 0.90, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0005500, 0.0004900]
    if fileName == "NGC2547":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.02, 0.11, 0.24, 0.37, 0.50, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0003900, 0.0002700]
    if fileName == "NGC3532":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.50, -0.05, 0.40, 0.85, 1.30, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0004100, 0.0003700]
    if fileName == "NGC6475":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.10, 0.20, 0.50, 0.80, 1.10, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0005900, 0.0005500]
    if fileName == "NGC6633":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.01, 0.42, 0.85, 1.28, 1.71, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0005130, 0.0002500]
    if fileName == "NGC7092":
        initial_guess = [r_i, 3.42, 5.50, 2.30, 2.55, -0.04, 0.31, 0.66, 1.01, 1.63, 0.20, 0.20, 0.20, 0.20,
                         mualpha_m_i, mudelta_m_i, 0.0005800, 0.0004800]
    if fileName == "Praesepe":
        print("nothing")

    bins = calculate_bins(bp_rp_list, 4)

    # Find number of stars in every bin (color bin)
    w = findWeightings(starlist, bins)

    print "Number of stars in cluster", len(starlist)

    # MLE constants
    # N: number of stars
    # mLim: magnitud limit of the mission
    # lmean: mean l
    # bmean: mean b
    constants_mle = {'N': len(starlist), 'mlim': 20.0, 'lmean': calculateMeanL(starlist, len(starlist)),
                     'bmean': calculateMeanB(starlist, len(starlist)), "w": w}

    # Calculates ML and errors
    results = MLE(starlist, constants_mle, initial_guess, bins)
    calculateErrors(starlist, constants_mle, results, bins)
