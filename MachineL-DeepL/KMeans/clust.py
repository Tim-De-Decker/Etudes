# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, rand, randint
from statistics import mode

class KMeans:

    def __init__(self, B, afficher=True):
        """
        Initialise un classificateur selon la méthode des K-moyennes.
        L'argument B est obligatoire : il peut s'agir d'un entier correspondant au nombre de classes désirées ou d'une liste de points.
        """
        self.B = B
        if afficher:
            print("Estimateur créé")

    
    def fit(self, X):
        """
        Se base sur l'ensemble de points donné en paramètre pour identifier des classes grâce au calcul de barycentres successifs.
        Les barycentres obtenus en fin d'algorithme sont contenus dans l'attribut .B
        """
        self.X = X
        X0 = np.array([x[0] for x in X])
        X1 = np.array([x[1] for x in X])

        # Si l'utilisateur renseigne un nombre de classes voulues plutôt que des points, 
        # on génère aléatoirement autant de centroïdes que nécessaire.
        # On délimite leur position en se basant sur les coordonnées des points les plus extrêmes.
        X0min, X0max, X1min, X1max = np.min(X0), np.max(X0), np.min(X1), np.max(X1)  
        if type(self.B) == int:
            self.B = [[uniform(X0min, X0max), uniform(X1min, X1max)] for _ in range(self.B)]
        
        classes_prec = randint(1, 4, len(X0))
        
        for _ in range(1000):
            
            # On crée un array contenant les distances à chaque barycentre (lignes) de chaque point de X (colonnes)
            dist = np.array([np.sqrt( (X0-b[0])**2 + (X1-b[1])**2 ) for b in self.B])
            
            # On récupère l'indice du barycentre le plus proche (soit la classe) pour chaque point
            classes = np.argmin(dist, 0)
            
            # Si le classement est identique au précédent, on stoppe la boucle
            if (classes == classes_prec).all():
                break
            
            for i in range(len(self.B)):
                
                # Pour chaque barycentre on retrouve les coordonnées des points du cluster correspondant
                clust = X0[classes == i], X1[classes == i]
                N = len(clust[0])
                
                if N > 0:
                    # Si le cluster contient des points, on calcule le nouveau barycentre
                    self.B[i] = [sum(clust[0])/N, sum(clust[1])/N]
                
                else:
                    # S'il est vide, on en "repioche" un au hasard
                    self.B[i] = [uniform(X0min, X0max), uniform(X1min, X1max)]
        
        self.classes = classes
 
    
    def predict(self, P):
        """Prédit la classe d'appartenance du point P après calcul des clusters par la méthode fit()"""
        dist = np.array([np.sqrt( (P[0]-b[0])**2 + (P[1]-b[1])**2 ) for b in self.B])
        classe = np.argmin(dist, 0)
        return classe
    
    def score(self, y):
        """
        Calcule un score correspondant au taux d'erreurs de classification en fonction des vraies classes des points donnés en arguments de fit()
        Les points mal classés sont enregistrés dans l'attribut .Xerr'
        """
        self.y = y
        err = {}
        self.Xerr = []
        for i in range(len(self.B)):
            true_y = y[self.classes==i]
            classe = mode(true_y)
            err[f"Vraie classe {classe}"] = np.count_nonzero(true_y != classe) / len(true_y)
            self.Xerr = self.Xerr + self.X[self.classes==i, :][true_y != classe].tolist()
        err['moyenne'] = sum(err.values()) / len(err)
        return err



# Autotest

if __name__ == '__main__':
    test = KMeans(2)
    autoX = np.array([[rand(), rand()] for _ in range(100)])
    test.fit(autoX)
    plt.figure(figsize=(10,8))
    plt.scatter(autoX[:,0], autoX[:,1], c=test.classes, cmap='copper', s=100)
    plt.scatter([b[0] for b in test.B], [b[1] for b in test.B], marker='*', c='red', s=300)
    plt.show()

