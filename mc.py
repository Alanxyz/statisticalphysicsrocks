import numpy as np
import matplotlib.pyplot as plt

# Physical variables

density = 0.20
D = 4

# Non-physical parameters

sidepartnum = 5
accratio = 0.3
iterations = 10_000  # recomendation: 1000 N
drmax = 1

partnum = sidepartnum ** D
boxlen = (partnum / density)**(1/D)
conf = np.array([
    (np.array([i, j, k, l]) + 0.5) * density**(-1/D)
    for i in range(sidepartnum)
    for j in range(sidepartnum)
    for k in range(sidepartnum)
    for l in range(sidepartnum)
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

def gravpot(r):
    return 1 / r**2

def hspot(r):
    return np.inf if r < 1 else 0

def ljpot(r):
    invr6 = (1 / r) ** 6
    return 4 * (invr6 ** 2 - invr6)

def pairpot(r1, r2):
    rvec = r2 - r1
    rvec -= boxlen * np.round(rvec / boxlen)
    r = np.linalg.norm(rvec)
    return ljpot(r) if r < boxlen / 2 else 0

def partener(n):
    r = conf[n]
    otherparts = np.delete(conf, n, axis=0)
    energies = np.apply_along_axis(lambda _: pairpot(_, r), 1, otherparts)
    return np.sum(energies)

def totener():
    energies = np.array([ partener(n) for n in range(partnum) ])
    return 0.5 * np.sum(energies)


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
    chain = []
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
            chain.append(energy)
            accepteds += 1
        else:
            conf[n] = initpos.copy()

        if tries % 100 == 0:
            ratio = accepteds / tries
            #adjustdr(ratio)
            print(f'{tries}\t{energy}\t{drmax}\t{ratio}')

    return chain

# Plots

def plot(conf):
    fig, axs = plt.subplots(4, 4, layout="constrained")
    pairs = [ (i, j) for i in range(4) for j in range(4) if j < i ]
    for pair in pairs:
        axs[pair[0], pair[1]].set_title(f'Plane {pair[0]}{pair[1]}')
        axs[pair[0], pair[1]].scatter(conf[:, pair[0]], conf[:, pair[1]])
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
chain = sampling(1000)
plot(conf)


