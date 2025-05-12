import numpy as np

# Physical variables

density = 0.20

# Non-physical parameters

sidepartnum = 7
accratio = 0.3
iterations = 10_000  # recomendation: 1000 N
drmax = 1

partnum = sidepartnum ** 3
boxlen = np.cbrt(partnum / density)
conf = np.array([
    (np.array([i, j, k]) + 0.5) * boxlen / sidepartnum
    for i in range(sidepartnum)
    for j in range(sidepartnum)
    for k in range(sidepartnum)
])

# Utils

def banner():
    print(f'''
____________________________________

            Monte-Carlo
____________________________________


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
    return np.sum(energies)


# Motion

def boundcond():
    conf[np.any(conf > boxlen, axis=1)] %= boxlen

def movepart(n):
    v = np.random.randn(3)
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

def sampling():
    chain = []

    energy = totener()
    j = 0
    for i in range(iterations):
        n = np.random.randint(0, partnum)
        initpos = conf[n].copy()
        initener = partener(n)
        movepart(n)
        newener = partener(n)

        diffener = newener - initener

        if np.exp(-diffener) > np.random.rand():
            j += 1
            energy += diffener
            #chain.append(conf.copy())
            chain.append(energy)
        else:
            conf[n] = initpos.copy()

        adjustdr(j / (i + 1))
        if i % 100 == 0:
            print(f'{i}\t{energy}\t{drmax}\t{j / (i + 1)}')

    return chain

# main

banner()
chain = sampling()
import matplotlib.pyplot as plt
from matplotlib import animation

def plot(chain):
    ax = plt.figure().add_subplot(projection='3d')
    x = chain[:, 0]
    y = chain[:, 1]
    z = chain[:, 2]
    ax.scatter(x, y, z)
    plt.show()

plt.plot(chain)
plt.show()

