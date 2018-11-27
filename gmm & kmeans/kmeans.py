import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
       '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
       assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
       np.random.seed(42)
       N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

       # DONOT CHANGE CODE ABOVE THIS LINE

       mu_idx = np.random.choice(N,self.n_cluster)
       mu = np.array([x[i] for i in mu_idx])
       distortion = 10 ** 10

       centroids = {}
       for i in range(self.n_cluster):
           centroids[i] = x[mu_idx[i]]

       assignment = np.zeros([N,self.n_cluster])
       # print('assignment ', assignment.shape, 'distortion ',distortion)

       membership = np.zeros([N, ])
       # for t in range(2):
       for t in range(self.max_iter):

           clusters = {}
           for i in range(self.n_cluster):
               clusters[i] = []

           # print('mu ', mu)
           dists = np.zeros((N, self.n_cluster))
           A = np.sum(x ** 2, axis=1).reshape(N, 1)
           B = np.sum(mu ** 2, axis=1).reshape(self.n_cluster, 1)
           AB = np.dot(x, np.transpose(mu))
           dists = np.sqrt(-2 * AB + A + np.transpose(B))
           # print('dists ',dists)

           membership = np.argmin(dists,axis=1)
           assignment = np.zeros((N, self.n_cluster))
           for i in range(N):
               assignment[i] = np.array([1 if j == membership[i] else 0 for j in range(self.n_cluster)])
               clusters[membership[i]].append(x[i])

           # b = np.argmin(dists,axis=1).reshape(-1,1)
           # c = np.array([i for i in range(len(dists))])
           # print('ass2 ', assignment3)

           # print('assignment ',assignment.shape)
           previous = dict(centroids)
           # print('previous',previous)
           # print('membership ',membership)
           # print('clusters ',clusters)

           for i in range(self.n_cluster):
               # print('clusters', clusters[i])
               # cluster = assignment[:,i]
               # print(t,' t : aggignment[:i]', assignment[:,i].shape,' cluster sum', np.sum(cluster))
               # if np.sum(cluster) == 0:
               #     print('    zero do not update, mu[i]', mu[i])
               #     continue
               # sumOfClusteri = np.array([cluster[i] * x[i] for i in range(N)])
               # mu[i] = np.sum(sumOfClusteri,axis=0,dtype='int')/np.sum(assignment[:,0])
               if len(clusters[i]) == 0:
                   print(t, ' len of cluster' ,len(clusters[i]))
               mu[i] = np.average(clusters[i],axis=0)
               centroids[i] = np.average(clusters[i],axis=0)
               # cen = np.average(np.array([x[j] for j in test[i]]),axis=0)
           # print('mu 2', mu)

           dists = np.zeros((N, self.n_cluster))
           A = np.sum(x ** 2, axis=1).reshape(N, 1)
           B = np.sum(mu ** 2, axis=1).reshape(self.n_cluster, 1)
           AB = np.dot(x, np.transpose(mu))
           C = np.sqrt(-2 * AB + A + np.transpose(B)) * assignment
           newDistortion = np.sum(C) / N

           # print(t, ' ', abs(distortion - newDistortion))
           if abs(distortion - newDistortion) < self.e:
               break
           else:
               distortion = newDistortion

       # membership = np.zeros([N,])
       # for i in range(N):
       #     label = np.sum(np.where(assignment[i] == 1),dtype='int')
       #     membership[i] = label

       # print('end:  mu ', mu, ' membership ', membership, ' t ',t)

       return mu, membership, t+1

       # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        mu_idx = np.random.choice(N, self.n_cluster)
        mu = np.array([x[i] for i in mu_idx])
        distortion = 10**10

        centroids_dict = {}
        membership = np.zeros([N, ])
        assignment = np.zeros([N, self.n_cluster])
        centroid_labels = np.zeros([self.n_cluster,])

        for i in range(self.n_cluster):
            centroids_dict[i] = x[mu_idx[i]]
            centroid_labels[i] = 0

        for t in range(self.max_iter):
            clusters = {}
            for i in range(self.n_cluster):
                clusters[i] = []

            dists = np.zeros((N, self.n_cluster))
            A = np.sum(x ** 2, axis=1).reshape(N, 1)
            B = np.sum(mu ** 2, axis=1).reshape(self.n_cluster, 1)
            AB = np.dot(x, np.transpose(mu))
            dists = np.sqrt(-2 * AB + A + np.transpose(B))
            membership = np.argmin(dists, axis=1)
            assignment = np.zeros((N, self.n_cluster))

            for i in range(N):
                # dists = [np.linalg.norm(x[i] - centroids_dict[k]) for k in centroids_dict]
                # idx = dists.index(min(dists))
                idx = np.argmin(dists[i])
                # assignment[i] = np.where(np.array(dists) == np.array(dists)[idx], 1, 0)
                # membership[i] = idx
                clusters[idx].append(i)

            previous = dict(centroids_dict)

            for i in range(self.n_cluster):
                centroids_dict[i] = np.average(np.array([x[j] for j in clusters[i]]), axis=0)
                y_labels = np.array([y[j] for j in clusters[i]])
                voteY = np.argmax(np.bincount(y_labels))
                # print(t, ' clusters i',clusters[i])
                centroid_labels[i] = voteY
                mu[i] = centroids_dict[i]
                # print('   ', t, '      voteY', voteY)

            dists = np.zeros((N, self.n_cluster))
            A = np.sum(x ** 2, axis=1).reshape(N, 1)
            B = np.sum(mu ** 2, axis=1).reshape(self.n_cluster, 1)
            AB = np.dot(x, np.transpose(mu))
            C = np.sqrt(-2 * AB + A + np.transpose(B)) * assignment
            newDistortion = np.sum(C) / N

            # newDistortion = 0
            # for i in range(N):
            #     newDistortion += np.sum(
            #         np.array([assignment[i][k] * np.linalg.norm(centroids_dict[k] - x[i]) for k in centroids_dict]))
            # newDistortion = newDistortion / N
            print('  abs(distortion - newDistortion) ', abs(distortion - newDistortion))

            if abs(distortion - newDistortion) < self.e:
                break
            else:
                distortion = newDistortion

        centroids = np.array([centroids_dict[i] for i in centroids_dict])

        print('centroid_labels ', centroid_labels.shape)
        print('centroids ', centroids.shape)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        print('x', x.shape)
        labels = np.zeros([N,])
        for i in range(N):
            dists = [np.linalg.norm(x[i] - muk) for muk in self.centroids]
            idx = dists.index(min(dists))
            labels[i] = self.centroid_labels[idx]

        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

