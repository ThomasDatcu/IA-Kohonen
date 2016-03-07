# coding: utf8
#!/usr/bin/env python
# ------------------------------------------------------------------------
# Carte de Kohonen
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------
# Implémentation de l'algorithme des cartes auto-organisatrices de Kohonen
# ------------------------------------------------------------------------
# Librairie de calcul matriciel
import numpy
# Librairie d'affichage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Pour lire les données MNIST
import gzip, cPickle

def neighborhood(pos_bmu,shape,width) :
    '''
    @summary: Fonction de voisinage
    @param pos_bmu: Position de l'unité gagnante
    @type pos_bmu: tuple
    @param shape: Taille de la SOM
    @type shape: tuple
    @param width: Largeur du voisinage
    @type width: float
    @return: La fonction de voisinage pour chaque unité de la SOM (numpy.array)
    '''

    '''
    On cherche à calculer ici la fonction de voisinage pour chaque neuronne.
    On prend une fonction gaussienne, ce qui nous donne la fonctione de
    voisinage pour *un* neuronne suivante :
    V(ij, ij*) = exp(-(||ij - ij*||)/(2*sigma**2))
    avec ij* les coordonnées du neuronne gagnant et sigma un paramètre constant.
    Sur le même principe que sur la fonction de distance entre Wij et X, on va
    calculer toutes les distance grâce à une seule opération de matrice en
    répliquant k,l fois les coordonnées ij.
    On construit X et Y de telle sorte que pour tout k,l :
        X[k,l] = i
        Y[k,l] = j
    On obtient donc la formule suivante:
        V = exp(-(X**2+Y**2)/2*sigma**2)
    '''

    X,Y = numpy.ogrid[0:shape[0],0:shape[1]]
    X -= pos_bmu[0]
    Y -= pos_bmu[1]
    return numpy.exp(-(X**2+Y**2)/(2.*width**2))

def distance(proto,inp):
    '''
    @summary: Fonction de distance
    @param proto: Prototypes de la SOM
    @type proto: numpy.array
    @param input: Entrée courante
    @type input: numpy.array
    @return: La distance entre l'entrée courante et le prototype pour chaque unité de la SOM (numpy.array)

    '''

    '''
    Soit proto ( alias W ) la matrice des poids de dimension de la
    dimension de l'entrée fois la dimention de l'ensemble des neuronnes.
    Ici, W est de dimension 4, soit W[i,j,k,l] avec :
        - i,j les coordonnées d'un neuronne dans une matrice des neuronnes
        - k,l les coordonnées d'une composante de l'entrée
    Soit inp ( alias X ) la matrice d'entrée (ici de dimension 2), soit X[k,l]
    La formule du calcul de la distance euclidienne entre Wij et X est la
    suivante : ||Wij - X|| = sqrt(sum((Wij-X)**2)) // Avec sum -> somme sur k,l
    Pour pouvoir calculer la distance pour tous les neuronnes à la fois (et
    donc se débarrasser des boucles sur i et j), on copie l'entrée pour
    chaque neuronne.
    On construit donc une telle matrice de telle sorte que, pour chaque i,j
        Y[i,j] = X
    La formule de calcul de distance pour *tous les neuronnes* est donc :
        ||W - Y|| = sqrt(sum((W-Y)**2)) // Avec sum -> somme sur k,l
    '''
    k = 2
    l = 2
    W = proto
    Y = inp[numpy.newaxis,numpy.newaxis,:]
    C = (W - Y)**2
    B = numpy.sum(C,axis=k)
    A = numpy.sum(B,axis=l)
    return numpy.sqrt(A)

class SOM:
    ''' Classe implémentant une carte de Kohonen. '''

    def __init__(self, *args):
        '''
        @summary: Création du réseau
        @param args: Liste de taille des couches
        @type args: tuple
        '''
        # Taille des couches
        self.shape = args
        # Label associé à chaque unité de la carte
        self.labeling = numpy.empty(self.shape[1])
        # Entrée de la carte
        self.inp = numpy.empty(self.shape[0])
        # Poids de la carte (prototypes)
        self.weights = numpy.empty(self.shape[1]+self.shape[0])
        # Initialisation des poids
        self.reset()

    def reset(self):
        '''
        @summary: Initialisation des poids entre -1 et 1
        '''
        self.weights = numpy.random.random(self.shape[1]+self.shape[0])*2.-1.

    def learn(self,train_samples,epochs,lrate,width,verbose=False):
        '''
        @summary: Apprentissage du modèle
        @param train_sample: Ensemble des données d'entraînement
        @type train_sample: dictionnaire ('input' est un numpy.array qui contient l'ensemble des vecteurs d'entrées, 'output' est un numpy.array qui contient l'ensemble des vecteurs de sortie correspondants)
        @param epochs: Nombre de pas d'apprentissage
        @type epochs: int
        @param lrate: Taux d'apprentissage
        @type lrate: float
        @param width: Largeur du voisinage
        @type width: float
        @param verbose: Indique si l'affichage est activé (utilisable uniquement avec des entrées à deux dimensions)
        @type verbose: boolean
        '''

        lrate0 = lrate
        width0 = width
        tau = epochs / (numpy.log(width0))

        # Initialisation de l'affichage
        if verbose:
            # Création d'une figure
            plt.figure()
            # Mode interactif
            plt.ion()
            # Affichage de la figure
            plt.show()
        for i in range(epochs):
            # Choix d'un exemple aléatoire
            n = numpy.random.randint(train_samples['input'].shape[0])
            self.inp[:,:] = train_samples['input'][n]
            # Mise à jour des poids (utiliser les fonctions neighborhood et distance)
            d = distance(self.weights, self.inp)
            bmu = numpy.argmin(d)
            pos_bmu = numpy.unravel_index(bmu,self.shape[1])
            v = neighborhood(pos_bmu,self.shape[1],width)
            self.weights = self.weights + lrate*v[:,:,numpy.newaxis,numpy.newaxis]*(self.inp[numpy.newaxis,numpy.newaxis,:,:]-self.weights)
            # Apprentissage de la labelisation (avec la méthode de votre choix)
            output = train_samples['output'][n]
            self.labeling[pos_bmu[0],pos_bmu[1]] = output

            #Modification des paramètres
            lrate = lrate0 * numpy.exp(-i/epochs)
            width = width0 * numpy.exp(-i/tau)

            # Mise à jour de l'affichage
            if verbose and i%100==0:
                # Effacement du contenu de la figure
                plt.clf()
                # Remplissage de la figure
                self.scatter_plot(True)
                # Affichage du contenu de la figure
                plt.draw()
        # Fin de l'affichage interactif
        if verbose:
            # Désactivation du mode interactif
            plt.ioff()

    def test(self,test_samples,verbose=True):
        '''
        @summary: Test du modèle
        @param test_samples: Ensemble des données de test
        @type test_samples: dictionnaire ('input' est un numpy.array qui contient l'ensemble des vecteurs d'entrées, 'output' est un numpy.array qui contient l'ensemble des vecteurs de sortie correspondants)
        @param verbose: Indique si l'affichage est activé
        @type verbose: boolean
        '''
        # Erreur quadratique moyenne
        error = 0.
        for i in range(test_samples['input'].shape[0]):
            # Calcul du label correspondant à l'exemple de test courant
            d = distance(self.weights, test_samples['input'][i])
            bmu = numpy.argmin(d)
            pos_bmu = numpy.unravel_index(bmu,self.shape[1])
            label = self.labeling[pos_bmu[0], pos_bmu[1]]
            # Mise à jour de l'erreur quadratique moyenne
            error += numpy.sum((label-test_samples['output'][i])**2)
            # Affichage des résultats
            if verbose:
              print 'entree \n', test_samples['input'][i], '\nsortie %.2f' % label, '(attendue %.2f)' % test_samples['output'][i]
        # Affichage de l'erreur quadratique moyenne
        print 'erreur quadratique moyenne ',error/test_samples['input'].shape[0]

    def scatter_plot(self,interactive=False):
        '''
        Affichage du réseau dans l'espace d'entrée (dans le cas d'entrée à deux dimensions et d'une carte avec une topologie de grille)
        @param interactive: Indique si l'affichage se fait en mode interactif
        @type interactive: boolean
        '''
        # Création de la figure
        if not interactive:
            plt.figure()
        # Récupération des poids
        w = self.weights
        # Affichage des poids
        plt.scatter(w[:,:,0].flatten(),w[:,:,1].flatten(),c='k')
        # Affichage de la grille
        for i in xrange(w.shape[0]):
            plt.plot(w[i,:,0],w[i,:,1],'k',linewidth=1.)
        for i in xrange(w.shape[1]):
            plt.plot(w[:,i,0],w[:,i,1],'k',linewidth=1.)
        # Modification des limites de l'affichage
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        # Affichage de la figure
        if not interactive:
            plt.show()

    def plot(self):
        '''
        Affichage des poids du réseau (matrice des poids)
        '''
        # Création de la figure
        plt.figure()
        # Récupération et redimensionnement de la matrice de poids
        w = self.weights
        # Affichage des poids dans un sous graphique (suivant sa position de la SOM)
        for i in xrange(w.shape[0]):
            for j in xrange(w.shape[1]):
                plt.subplot(w.shape[0],w.shape[1],i*w.shape[1]+j+1)
                plt.imshow(w[i,j,:],interpolation='nearest',vmin=-1,vmax=1,cmap='binary')
                plt.xticks([])
                plt.yticks([])
        # Affichage de la figure
        plt.show()


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    verbose = False

    # Création d'un réseau avec une entrée (2,1) et une carte (10,10)
    #network = SOM((2,1),(10,10))
    network = SOM((28,28),(10,10))
    # Exemple 1 :
    # -------------------------------------------------------------------------
    # # Création des données d'apprentissage
    # n = 1000
    # train_input = numpy.random.random((n,2,1))*2.-1.
    # train_output = numpy.empty((n,1))
    # train_output[numpy.where(numpy.logical_and(train_input[:,0]<0,train_input[:,1]<0))] = 1.
    # train_output[numpy.where(numpy.logical_and(train_input[:,0]<0,train_input[:,1]>0))] = 2.
    # train_output[numpy.where(numpy.logical_and(train_input[:,0]>0,train_input[:,1]<0))] = 3.
    # train_output[numpy.where(numpy.logical_and(train_input[:,0]>0,train_input[:,1]>0))] = 4.
    # train = {'input':train_input,'output':train_output}
    #
    # # Création des données de test
    # n = 100
    # test_input = numpy.random.random((n,2,1))*2.-1.
    # test_output = numpy.empty((n,1))
    # test_output[numpy.where(numpy.logical_and(test_input[:,0]<0,test_input[:,1]<0))] = 1.
    # test_output[numpy.where(numpy.logical_and(test_input[:,0]<0,test_input[:,1]>0))] = 2.
    # test_output[numpy.where(numpy.logical_and(test_input[:,0]>0,test_input[:,1]<0))] = 3.
    # test_output[numpy.where(numpy.logical_and(test_input[:,0]>0,test_input[:,1]>0))] = 4.
    # test = {'input':test_input,'output':test_output}

    # Pour les données mnist (faire attention où est situé le fichier)
    data = cPickle.load(gzip.open('mnist.pkl.gz'))
    train = {'input':data[0][0].reshape(56000,28,28),'output':numpy.argmax(data[0][1],axis=1)}
    test = {'input':data[2][0].reshape(7000,28,28),'output':numpy.argmax(data[2][1],axis=1)}

    # Initialisation du réseau
    network.reset()
    # Performance du réseau avant apprentissage
    # NB Uniquement pertinent si la labelisation est faite
    print "Avant apprentissage"
    network.test(test,verbose)
    # Affichage du réseau dans l'espace d'entrée (puisqu'il est à deux dimensions)
    # network.scatter_plot()
    # Affichage des poids du réseau
    network.plot()
    # Apprentissage du réseau
    epochs = 15000 #30000
    lrate = 0.35 #0.02
    width = 4 #1.5

    network.learn(train,epochs,lrate,width,verbose)
    # Performance du réseau après apprentissage
    # NB Uniquement pertinent si la labelisation est faite
    print "Apres apprentissage"
    network.test(test,verbose)
    # Affichage du réseau dans l'espace d'entrée (puisqu'il est à deux dimensions)
    # network.scatter_plot()
    # Affichage des poids du réseau
    network.plot()
