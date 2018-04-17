"""
@author: M. Palmer

Module that contains the transformations needed to determine MLE (please refer to Xavier Luri Carrascoso PhD thesis (1993)
Un Nuevo metodo de maxima verosimilitud para la determinacion de magnitudes absolutas
"""
import sys
from math import pow, pi, sqrt, log, log10, exp, sin, cos, atan2, asin
from multiprocessing import Process

from scipy import interpolate

# Options: Cluster, kinematics, catalogue
# TODO: This needs some refactoring
cluster="Pleiades" # Pleiades or Hyades
kinimatics=False   # True or False
cat="New"          # New or Old

# If cluster is Pleiades or Hyades (age)
if cluster=="Pleiades": AgCluster = 0.0975
else: AgCluster =0.0
print "Cluster Extinction is", AgCluster
k=4.74
Dg=0.4782
Ag=4.9291


# Transformations (refer to Xavi's PhD thesis)
# TODO: pending documentation
def Fv(p,s,r0): return (K3(p,s,r0)) *exp ((Y(p,s,r0)**2/(4*X(p,s,r0))) - Z(p,s,r0))   # P is Params array, S is star data array
def K3(p,s,r0): return sqrt(- pi**3 / (J2(p,s,r0)*E1(p,s,r0)*X(p,s,r0) ))
def X(p,s,r0):  return (D1(p,s,r0)**2/(4*E1(p,s,r0)))-F1(p,s,r0)
def Y(p,s,r0):  return ((B1(p,s,r0)*D1(p,s,r0))/(2*E1(p,s,r0)))-C1(p,s,r0)
def Z(p,s,r0):  return ( pow(B1(p,s,r0),2)/(4*E1(p,s,r0)))-A1(p,s,r0)
def A1(p,s,r0): return ( pow(A2(p,s,r0),2)/(4*J2(p,s,r0)))-D2(p,s,r0)
def B1(p,s,r0): return ((A2(p,s,r0)*B2(p,s,r0))/(2*J2(p,s,r0))) - E2(p,s,r0)
def C1(p,s,r0): return ((A2(p,s,r0)*C2(p,s,r0))/(2*J2(p,s,r0))) - F2(p,s,r0)
def D1(p,s,r0): return ((B2(p,s,r0)*C2(p,s,r0))/(2*J2(p,s,r0))) - G2(p,s,r0)
def E1(p,s,r0): return ( pow(B2(p,s,r0),2)/(4*J2(p,s,r0)))-H2(p,s,r0)
def F1(p,s,r0): return ( pow(C2(p,s,r0),2)/(4*J2(p,s,r0)))-I2(p,s,r0)

def A2(p,s,r0): return -(s['muAlpha']/ pow(s['epsMuAlpha'],2)) - ((  ((a1(s)*p['U'])/ pow(p['sigmaU'],2)) + ((a2(s)*p['V'])/ pow(p['sigmaV'],2)) + ((a3(s)*p['W'])/ pow(p['sigmaW'],2))  ) *r0)
def B2(p,s,r0): return (  ((a1(s)*b1(s))/ pow(p['sigmaU'],2)) + ((a2(s)*b2(s))/ pow(p['sigmaV'],2)) + ((a3(s)*b3(s))/ pow(p['sigmaW'],2))  ) *  pow(r0,2)
def C2(p,s,r0): return (  ((a1(s)*c1(s))/ pow(p['sigmaU'],2)) + ((a2(s)*c2(s))/ pow(p['sigmaV'],2)) + ((a3(s)*c3(s))/ pow(p['sigmaW'],2))  ) *r0
def D2(p,s,r0): return 0.5 * ((pow(s['muAlpha'],2)/pow(s['epsMuAlpha'],2)) + (pow(s['muDelta'],2)/pow(s['epsMuDelta'],2)) + (pow(s['vr'],2)/pow(s['epsVr'],2)) + (pow(p['U'],2)/pow(p['sigmaU'],2)) + (pow(p['V'],2)/pow(p['sigmaV'],2)) + (pow(p['W'],2)/pow(p['sigmaW'],2)))
def E2(p,s,r0): return -(s['muDelta']/ pow(s['epsMuDelta'],2)) - (( ((b1(s)*p['U'])/pow(p['sigmaU'],2)) + ((b2(s)*p['V'])/pow(p['sigmaV'],2)) + ((b3(s)*p['W'])/pow(p['sigmaW'],2))) *r0)
def F2(p,s,r0): return -(s['vr']/ pow(s['epsVr'],2)) - ( ((c1(s)*p['U'])/pow(p['sigmaU'],2)) + ((c2(s)*p['V'])/pow(p['sigmaV'],2)) + ((c3(s)*p['W'])/pow(p['sigmaW'],2))  )
def G2(p,s,r0): return (  ((b1(s)*c1(s))/pow(p['sigmaU'],2)) + ((b2(s)*c2(s))/pow(p['sigmaV'],2)) + ((b3(s)*c3(s))/pow(p['sigmaW'],2)) ) *r0
def H2(p,s,r0): return (1.0/(2*pow(s['epsMuDelta'],2))) + 0.5*(  (( pow(b1(s),2))/( pow(p['sigmaU'],2)))+(( pow(b2(s),2))/( pow(p['sigmaV'],2)))+(( pow(b3(s),2))/( pow(p['sigmaW'],2))))* pow(r0,2)
def I2(p,s,r0): return (1.0/(2*pow(s['epsVr'],2)))  + 0.5*(  (( pow(c1(s),2))/( pow(p['sigmaU'],2)))+(( pow(c2(s),2))/( pow(p['sigmaV'],2)))+(( pow(c3(s),2))/( pow(p['sigmaW'],2))))
def J2(p,s,r0): return (1.0/(2*pow(s['epsMuAlpha'],2))) + 0.5*(  (( pow(a1(s),2))/( pow(p['sigmaU'],2)))+(( pow(a2(s),2))/( pow(p['sigmaV'],2)))+(( pow(a3(s),2))/( pow(p['sigmaW'],2))))* pow(r0,2)

def a1(s): return k*(s['sfi']*s['cl']*s['sb']-s['cfi']*s['sl'])
def a2(s): return k*(s['sfi']*s['sl']*s['sb']+s['cfi']*s['cl'])
def a3(s): return -k*s['sfi']*s['cb']
def b1(s): return -k*(s['cfi']*s['cl']*s['sb']+s['sfi']*s['sl'])
def b2(s): return -k*(s['cfi']*s['sl']*s['sb']-s['sfi']*s['cl'])
def b3(s): return k*s['cfi']*s['cb']
def c1(s): return s['cl']*s['cb']
def c2(s): return s['sl']*s['cb']
def c3(s): return s['sb']

def getU(a,s): return a1(a)*s['muAlpha']/s['pi']  +  b1(a)*s['muDelta']/s['pi']  +  c1(a)*s['vr']
def getV(a,s): return a2(a)*s['muAlpha']/s['pi']  +  b2(a)*s['muDelta']/s['pi']  +  c2(a)*s['vr']
def getW(a,s): return a3(a)*s['muAlpha']/s['pi']  +  b3(a)*s['muDelta']/s['pi']  +  c3(a)*s['vr']

def calculateMeanL(slist, N):
    """
    Calculates the mean l coordinate
    :param slist: star list
    :param N: number of stars in the cluster
    :return: mean l coordinate
    """
    return atan2(sum([item['sinl'] for item in slist]) / N, sum([item['cosl'] for item in slist]) / N)

def calculateMeanB(slist, N):
    """
    Calculates the mean b coordinate

    :param slist: star list
    :param N: number of stars in the cluster
    :return: mean b coordinate
    """

    return atan2(sum([item['sinb'] for item in slist]) / N, sum([item['cosb'] for item in slist]) / N)


def calculateMeanR(slist, N):
    '''
    Calculates the mean R
    :param slist:
    :param N:
    :return:
    '''
    r = 0
    for item in slist:
        r = r + (1.0 / item['pi'])
    rmean = r / N
    return rmean

def Fi(a,l,b):
    '''

    :param a:
    :param l:
    :param b:
    :return:
    '''
    x=cos(Dg)*sin(getDelta(a,l,b))*sin(getAlpha(a,l,b)-Ag)+sin(Dg)*cos(getDelta(a,l,b))
    y=cos(Dg)*cos(getAlpha(a,l,b)-Ag)
    return atan2(y,x)

def getA(l,b):
    '''

    :param l:
    :param b:
    :return:
    '''
    a={'sb':sin(b),'cb':cos(b),'sl':sin(l),'cl':cos(l)}
    a.update({'cfi':cos(Fi(a,l,b)),'sfi':sin(Fi(a,l,b))})
    return a

def getDelta(a,l,b): return asin(a['cb']*sin(l-33*pi/180)*sin(62.6*pi/180)+a['sb']*cos(62.6*pi/180))

def getAlpha(a,l,b):
    '''

    :param a:
    :param l:
    :param b:
    :return:
    '''
    t1=(a['cb']*sin(l-33*pi/180)*cos(62.6*pi/180))-(a['sb']*sin(62.6*pi/180))
    t2=a['cb']*cos(l-33*pi/180)
    return atan2(t1,t2) +Ag

def getVr(a,U,V,W): return (W- U*a3(a)/a1(a) - (V-U*a2(a)/a1(a)) * (b3(a)-b1(a)*a3(a)/a1(a)) / (b2(a)-b1(a)*a2(a)/a1(a))) /(( c3(a)-c1(a)*a3(a)/a1(a) )-(c2(a)-c1(a)*a2(a)/a1(a)) * (b3(a)-b1(a)*a3(a)/a1(a)) / (b2(a)-b1(a)*a2(a)/a1(a)))
def getMuDelta(a,V,U,vr,r): return ((V-U*a2(a)/a1(a) ) - (( c2(a)-c1(a)*a2(a)/a1(a) ) * vr)) / r / ( b2(a)-b1(a)*a2(a)/a1(a) )
def getMuAlpha(a,U,vr,r,muDelta): return  (U-r*b1(a)*muDelta-c1(a)*vr)/r/a1(a)

def logD(D):
    '''
    Safest way of calculating the log (if the probability from ML is zero is not possible to calculate the log, although
    if the probability is zero it makes mathematical sense)
    If the probability is zero (e.g. due to rounding), it assigns a very large negative number

    :param D:
    :param star:
    :return:
    '''
    if D > 0:     D =  log(D)
    elif D == 0:  D = -10000
    elif D < 0:
        print "Error: negative D"
        sys.exit()
    return D

def transformCoordinates(s,c):
    '''

    :param s:
    :param c:
    :return:
    '''
    s['lprime'] = s['l'] - c['lmean']
    if s['lprime'] < 0: s['lprime'] = 2 *  pi + s['lprime']
    if (- pi / 2) - 0.2 > c['bmean'] or ( pi / 2) - 0.2 < c['bmean']  : print "WARNING: Close to anticenter... check coordinate rotation", c['bmean'], s['l'], s['b']
    s['bprime'] = s['b'] - c['bmean']
    s.update({'sb':sin(s['b']),'cb':cos(s['b']),'sl':sin(s['l']),'cl':cos(s['l']),'sbprime':sin(s['bprime']),'cbprime':cos(s['bprime']),'slprime':sin(s['lprime']),'clprime':cos(s['lprime'])})
    a=getA(s['l'],s['b'])
    s.update({'cfi':cos(Fi(a,s['l'],s['b'])),'sfi':sin(Fi(a,s['l'],s['b']))})
    return s


def getParameters(x):
    '''
    Returns a dictionary for the parameters with their labels, it also computes R^2 (the distance to the cluster)

    :param x:
    :return:
    '''
    if kinimatics==True:
        labels =  ['R','sS1','sS2','sS3','sS4','x1','x2','x3','x4','x5','sM1','sM2','sM3','sM4','U','V','W','sigmaUVW']
        p=dict(zip(labels, x))
        p.update({'R^2':p['R']**2,"sigmaU":p['sigmaUVW'],"sigmaV":p['sigmaUVW'],"sigmaW":p['sigmaUVW'] })
    elif kinimatics==False:
        labels =  ['R','sS1','sS2','sS3','sS4','x1','x2','x3','x4','x5','sM1','sM2','sM3','sM4','muAlphaMean','muDeltaMean','sigmaMuAlpha','sigmaMuDelta']
        p=dict(zip(labels, x))
        p.update({'R^2':p['R']**2})
    return p

def checkParamsAreValid(p):
    '''

    :param p:
    :return:
    '''
    if kinimatics==True:
        if min(p['sS1'],p['sS2'],p['sS3'],p['sS4'],p['sM1'],p['sM2'],p['sM3'],p['sM4'],p['R'],p['sigmaUVW'])<=0:
            return False
        else: return True
    elif kinimatics==False:
        if min(p['sS1'],p['sS2'],p['sS3'],p['sS4'],p['sM1'],p['sM2'],p['sM3'],p['sM4'],p['R'],p['sigmaMuAlpha'],p['sigmaMuDelta'])<=0:
            return False
        else: return True


def startProcess(procs, target, args):
    '''

    :param procs:
    :param target:
    :param args:
    :return:
    '''
    thisProcess = Process(target=target, args=args)
    procs.append(thisProcess)
    thisProcess.start()
    return procs

def getSigmaSSigmaMforC(p, minColour, maxColour, bins):
    '''

    :param p:
    :param minColour:
    :param maxColour:
    :return:
    '''
    for i in xrange(0,len(bins),1):
        if minColour == bins[i][0] and maxColour == bins[i][1]:
            sigmaS = p['sS%s' % str(i+1)]
            sigmaM = p['sM%s' % str(i+1)]
    return sigmaS, sigmaM

def getSigmaSSigmaMforD(p, star, bins):
    '''

    :param p:
    :param star:
    :return:
    '''
    sigmaS=None
    for i in xrange(0,len(bins),1):
        if star['bp-rp'] >= bins[i][0] and star['bp-rp'] < bins[i][1]:
            sigmaS = p['sS%s' % str(i+1)]
            sigmaM = p['sM%s' % str(i+1)]
    return sigmaS, sigmaM

def getCminCmaxforD(star, bins):
    '''

    :param star:
    :return:
    '''
    for thisBin in bins:
        if star['bp-rp'] >= thisBin[0] and star['bp-rp'] < thisBin[1]:
            return thisBin[0],thisBin[1]

def getCminCmaxforC(minColour, maxColour, bins):
    '''

    :param minColour:
    :param maxColour:
    :return:
    '''
    for thisBin in bins:
        if minColour == thisBin[0] and maxColour == thisBin[1]:
            return thisBin[0],thisBin[1]

def getWeight(c,minColour,maxColour, bins):
    '''

    :param c:
    :param minColour:
    :param maxColour:
    :return:
    '''
    for i in xrange(0,len(bins),1):
        if minColour == bins[i][0] and maxColour == bins[i][1]:
            w=c['w']
            return w[i]

# Distributions for magnitud, space, velocity, dispersion in distance and proper motions
def Fmag(star,r0,spline,sigmaM): return exp(-0.5 *  pow((star['mg'] - (5*log10(r0)) + 5 - AgCluster - (interpolate.splev(star['bp-rp'],spline))) / sigmaM, 2))
def Fspace(star,r0,p,sigmaS): return exp (-0.5 * (p['R^2'] + pow(r0,2) - (2 * r0 * p['R'] *  star['cbprime'] * star['clprime'])) / (sigmaS**2) )
#def Fvel(star,muAlpha0,muDelta0,p): return exp (-0.5 *  pow((muAlpha0 - p['muAlphaMean']) / p['sigmaMuAlpha'], 2)) * exp (-0.5 *  pow((muDelta0 - p['muDeltaMean']) / p['sigmaMuDelta'], 2))
def Fvel(muAlpha0,muDelta0,p_mualpha, p_sigma_mualpha, p_mudelta, p_sigma_mudelta): return exp (-0.5 *  pow((muAlpha0 - p_mualpha) / p_sigma_mualpha, 2)) * exp (-0.5 *  pow((muDelta0 - p_mudelta) / p_sigma_mudelta, 2))


def jacobian(star,r0): return pow(r0,2)* star['cbprime']
def epsPi(star,r0): return exp (-0.5 *  pow((star['pi'] - (1.0 / r0)) / star['epsPi'], 2))
def epsMuAlpha(star,muAlpha0): return exp (-0.5 *  pow((star['muAlpha'] -  muAlpha0) / star['epsMuAlpha'], 2))
def epsMuDelta(star,muDelta0): return exp (-0.5 *  pow((star['muDelta'] -  muDelta0) / star['epsMuDelta'], 2))


def getSpline(bins,p):
    '''
    Spline function which gives the isochrone
    :param bins:
    :param p:
    :return:
    '''
    y=[p['x1']]
    x=[bins[0][0]]
    for i in bins:
        x.append(i[1])
    for i in xrange(0,len(bins),1):
        y.append(p['x%s' %str(i+2)])
    return interpolate.splrep(x,y)


def findWeightings(starlist, bins):
    '''
    Find the relative weightings to be applied to each bin,
    computed from number of stars inside each color bin divided by the total number of stars
    :param starlist:
    :return:
    '''
    w=[0,0,0,0]
    for star in starlist:
        for i in xrange(0,len(bins)):
            if star['bp-rp']<bins[i][1] and star['bp-rp'] > bins[i][0]: 
                w[i]=w[i]+1
    print "number per bin",w
    for i in xrange(0,len(w)):
        w[i]=float(w[i])/len(starlist)
    print "weightings",w
    return w

