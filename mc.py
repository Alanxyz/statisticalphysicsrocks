import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

# Physical variables

D = 4
packfrac = 0.30

# Non-physical parameters

sidepartnum = 5

accratio = 0.3
drmax = 1

density = packfrac / (np.pi**(D/2) * (1/2)**D / math.gamma(D/2 + 1))
partnum = sidepartnum ** D
boxlen = (partnum / density)**(1/D)
conf = np.array([
    (np.array(index) + 0.5) * density**(-1/D)
    for index in itertools.product(range(sidepartnum), repeat=D)
])

# Utils

def banner():
    print(f'''
____________________________________

            Monte-Carlo
____________________________________


D = {D}
N = {partnum}
d = {density}
L = {boxlen}

''')

# Energy

def partener(n):
    rn = conf[n]
    rvec = conf - rn
    rvec = rvec[np.arange(partnum) != n]
    rvec -= boxlen * np.round(rvec / boxlen)
    r = np.linalg.norm(rvec, axis=1)

    u = np.zeros(partnum - 1)
    dl = 50
    dT = 1.4737
    mask = r <= dl / (dl - 1)
    r = r[mask]
    u[mask] = ((dl * ( dl / (dl - 1))**(dl - 1))  / dT) * ((1 / r)**dl - (1 / r)**(dl - 1)) + 1 / dT

    return np.sum(u)

def totener():
    energies = np.array([ partener(n) for n in range(partnum) ])
    return .5 * np.sum(energies)


# Motion

def boundcond():
    conf[np.any(conf > boxlen, axis=1)] %= boxlen

def movepart(n):
    v = np.random.randn(D)
    u = v / np.linalg.norm(v)
    dr = u * np.random.rand() * drmax
    conf[n] += dr
    conf[n] %= boxlen

# MonteCarlo sex

def mloglikelihood():
    pass

def mlogprior():
    pass

def adjustdr(ratio):
    global drmax
    drmax *= 1.05 if ratio > accratio else 0.95

def sampling(iterations):
    enerchain = []
    confchain = []

    energy = totener()
    accepteds = 0
    for tries in range(1, iterations + 1):
        n = np.random.randint(partnum)
        initpos = conf[n].copy()
        initener = partener(n)
        movepart(n)
        finalener = partener(n)

        diffener = finalener - initener

        if diffener < 0 or np.exp(-diffener) > np.random.rand():
            energy += diffener
            enerchain.append(energy)
            confchain.append(conf)
            accepteds += 1
            if accepteds % partnum == 1:
                enerchain.append(energy)
                confchain.append(conf.copy())
        else:
            conf[n] = initpos.copy()

        ratio = accepteds / tries
        adjustdr(ratio)

        if tries % 100 == 0:
            print(f'{tries}\t{energy}\t{drmax}\t{ratio}')

    return np.array(enerchain), np.array(confchain)

# Plots

def plothist(data, ax):
    counts, bins = np.histogram(data, bins=20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax.bar(bin_centers, counts, width=bin_width, edgecolor='black')

def plot(conf):
    fig, axs = plt.subplots(D, D, layout="constrained") 
    pairs = [ (i, j) for i in range(D) for j in range(D) if j < i ]
    for pair in pairs:
        axs[pair[0], pair[1]].set_title(f'Plane {pair[0]}{pair[1]}')
        axs[pair[0], pair[1]].scatter(
            conf[:, pair[0]],
            conf[:, pair[1]],
            marker='.',
            alpha=0.5
            )
    for i in range(D):
        plothist(conf[:, i], axs[i, i])
    plt.show()

def plotheat(conf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = conf[:, 0]
    y = conf[:, 1]
    z = conf[:, 2]
    c = conf[:, 3]

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()

# main

banner()
enerchain, confchain = sampling(100_000)
