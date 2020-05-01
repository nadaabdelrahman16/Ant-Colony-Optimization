import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

citiesCoordinates = pd.read_csv("TSPdata1.csv", header=None, names=['x', 'y'])
citiesCoordinates = np.array(citiesCoordinates)
nCities = citiesCoordinates.shape[0]


def calculate_distance(citiesCoordinates, nCities):
    # Use this function to Calculate the Euclidean distance between every two cities
    # citiesCoordinates : population size
    # nCities : number of cities in the network
    #distMat : Matrix n*n where store Euclidean distance between every two cities
    distMat = np.zeros((nCities, nCities), dtype=float)
    for i in range(nCities):
        for j in range(nCities):
            if i == j:
                # when destination city and current city are same so dist equal to zero
                distMat[i,j]=0
            else:
                dist = 0
                for k in range(2):
                    dist += (citiesCoordinates[i, k] - citiesCoordinates[j, k]) ** 2
                distMat[i,j] = dist ** 0.5
    return distMat



def calculate_eta(distMat, nCities):
    # Use this function to store the reciprocal of the distances , where eta(i,j) = 1/dist(i,j)
    # nCities : number of cities in the network
    # distMat : Matrix n*n where store Euclidean distance between every two cities
    # modDist : Matrix equal to distMat
    # eta : Matrix where store the reciprocal of the distances
    modDist = distMat[:, :]
    # to ignore math error to edges with zero dist
    for i in range(nCities):
        modDist[i, i] = np.inf
    eta = np.true_divide(1, modDist)
    return eta



def getLnn(start, end,distMat):
    # use this Function to initialize phermone initial point using nearest neighbor technic
    # start: start the path at any city representing the nest of the ants
    # end: end the path at any city
    # distMat: Matrix n*n where store Euclidean distance between every two cities
    #LnnPath :
    LnnPath = np.zeros(1, dtype=int)
    LnnPath[0] = start

    for i in range(nCities - 1):
     # assigning 1000000000.0 (as Big number) to ignore back to this city again
        distMat[start, start] = 1000000000
        distmat = distMat[start]
        nextNode = distmat.argmin()

        while (nextNode in LnnPath):
            distMat[start, nextNode] = 10000000000000
            distmat = distMat[start]
            nextNode = distmat.argmin()
        if (nextNode == end):
            LnnPath = np.append(LnnPath, end)
            break
        else:
            LnnPath = np.append(LnnPath, nextNode)
            start = nextNode

    return LnnPath


def calculate_tourlen(path, distMat):
    #use this function to calculate the length of the tour length produced by the nearest neighbor ant
    # distMat:Matrix n*n where store Euclidean distance between every two cities
    tourLen = 0.0
    for i in range(len(path) - 1):
        tourLen += distMat[path[i], path[i + 1]]
    tourLen += distMat[path[-1], path[0]]
    return tourLen




def calcProb(i, alpha, beta,eta,distMat,tau):
    #use this function to calculate probability as an ant will move from city to other city with probability
    # alpha is a parameter to control the influence of tau[i,l]
    # beta is a parameter to control the influence of eta[i,l]
    mult = np.zeros(nCities , dtype=float)

    p = np.zeros(nCities , dtype=float)
    for l in range(nCities):
        mult[l] = ((tau[i, l]) ** alpha) * ((eta[i, l]) ** beta) * (1 / distMat[i, l])
    mult[i] = 0.0
    for j in range(nCities):
        p[j] = mult[j] / (np.sum(mult))
    p[i] = 0.0
    return p



def getpath(start, alpha, beta,eta,distMat,tau):
    # Use this function to get path for ant by appling a state transition rule to incrementally build a solution
    path = np.zeros(1, dtype=int)
    path[0] = start
    for i in range(nCities-1):
        prob = calcProb(start, alpha, beta,eta,distMat,tau)

        nextNode = prob.argmax()

        while (nextNode in path):
            prob[nextNode] = 0
            nextNode = np.argmax(prob)
        path = np.append(path, nextNode)
        start = nextNode

    return path



def pheromone_Evaporate(tau):
    # p: is a parameter that regulates the pheromone evaporation generated randomly
    # tau: is the set of all pheromone values
    for i in range(len(tau) - 1):
        for j in range(len(tau) - 1):
            p = np.random.random()
            tau[i, j] = (1 - p) * tau[i, j]
    return tau



def update_Pheromoneamount(path,distMat,tau):
    #use this function to update phermone
    for i in range(len(path) - 1):
        tau[path[i], path[i + 1]] += 1 / calculate_tourlen(path, distMat)
    return tau


def ACO(numofItr, AntsNum,alpha,beta):
    # end: representing the food source

    bestlen = 10000
    nCities=citiesCoordinates.shape[0]
    best_paths_len = np.zeros(numofItr)
    distMat = calculate_distance(citiesCoordinates, nCities)
    best_path = np.zeros((numofItr, nCities),dtype=int)
    sizes = np.zeros(AntsNum, dtype=int)
    lnn = getLnn(0, nCities-1,distMat)
    Lnn = calculate_tourlen(lnn, distMat)
    tau = np.zeros((nCities, nCities))
    for i in range(nCities ):
        for j in range(nCities ):
            tau[i, j] = 1 / (nCities* Lnn)
    for t in range(numofItr):
        paths =np.zeros((AntsNum, nCities), dtype=int)
        paths_len=np.zeros(AntsNum)
        for k in range(AntsNum):
            #start representing the nest of the ants and Each ant is positioned on a starting node
            start = np.random.randint(0, nCities - 1)
            eta=calculate_eta(distMat,nCities)
            path = getpath(start, alpha, beta,eta,distMat,tau)
            paths[k]=path
        for a in range(AntsNum):
            paths_len[a]=calculate_tourlen(paths[a], distMat)
        best_path_index=np.argmin(paths_len)
        # best path for all Ants in this iteration
        best_path[t]=paths[best_path_index]
        tau = pheromone_Evaporate(tau)
        for k in range(AntsNum):
            tau = update_Pheromoneamount(paths[0:(sizes[k]) - 1],distMat,tau)
            paths = paths[(sizes[k]):]
    # shortest_tour: best path for all iteration
    for a in range(numofItr):
        best_paths_len[a] = calculate_tourlen(best_path[a], distMat)
    shortest_tour=best_path[np.argmin(best_paths_len)]

    return shortest_tour

path=ACO(50,AntsNum=20,alpha=0.1,beta=2)
print(path)
# Plot the location of the cities and the shortest tour you found
G = nx.Graph()
G = nx.DiGraph()
for i in range(nCities):
   G.add_node(i)
edges=np.zeros((nCities,2),dtype=float)
edge=np.zeros((nCities-1,2), dtype=int)
for i in range (nCities):
    a=np.zeros((2),dtype=int)
    a[0] =path[i]
    if(i+1==nCities):
        break
    else:
     a[1]=path[i+1]
     edge[i]=a
G.add_edges_from(edge)
plt.figure(figsize =(20, 20))
nx.draw_networkx(G, with_label = True, node_color ='green')
plt.show()
