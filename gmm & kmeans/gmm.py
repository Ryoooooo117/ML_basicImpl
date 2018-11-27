import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE

            k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, membership, _ = k_means.fit(x)

            assignment = np.array([[1 if j == membership[i] else 0 for j in range(self.n_cluster)] for i in range(len(membership))])
            # print('ass', assignment.shape)

            # self.variances = np.array([ np.sum([assignment[n] * np.dot((x[n]-self.means[k]).T,x[n]-self.means[k]) / np.sum(assignment[:, k]) for n in range(N)]) for k in range(self.n_cluster)])

            self.variances = np.zeros([self.n_cluster,D,D])
            for k in range(self.n_cluster):
                sumCov = np.zeros([D,D])
                for n in range(N):
                    if assignment[n][k] == 1:
                        cov = np.dot(np.array(x[n]-self.means[k]).reshape(D,1), np.array(x[n]-self.means[k]).reshape((1,D)))
                        sumCov += cov
                N_k = np.sum(assignment[:, k])
                self.variances[k] = sumCov/N_k

            self.pi_k = np.array([np.sum(assignment[:, k])/N for k in range(self.n_cluster)])

            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE

            self.means = np.random.rand(self.n_cluster,D)
            self.variances = np.zeros([self.n_cluster, D, D])
            for k in range(self.n_cluster):
                self.variances[k] = np.identity(D)
            self.pi_k = np.array([1/self.n_cluster for k in range(self.n_cluster)])

            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE

        # print('\r\nx shape', x.shape, '\r\nself.means', self.means, '\r\nself.variances ', self.variances.shape, ' \r\npik', self.pi_k)

        ll = self.compute_log_likelihood(x)
        # print('ll ', ll, ' mean', self.means[0])

        K = self.n_cluster

        for t in range(self.max_iter):

            # E-Step:
            responsibilities = np.zeros([N, K])
            gaus_pdf = {}
            for k in range(K):
                gaus_pdf[k] = self.Gaussian_pdf(self.means[k], self.variances[k])

            pdf = np.zeros([N,K])
            for i in range(N):
                pdf_i = np.zeros([K,])
                resp = np.zeros([K,])
                for k in range(K):
                    pdf_i[k] = gaus_pdf[k].getLikelihood(x[i])
                    resp[k] = self.pi_k[k] * pdf_i[k]
                pdf[i] = pdf_i
                responsibilities[i] = resp / np.sum(resp)
            # print('respon ',responsibilities.shape)
            # print('sum res',np.sum(responsibilities,axis=0))

            # M-Step:
            self.means = np.array([np.sum([responsibilities[i][k] * x[i] for i in range(N)],axis=0) / np.sum(responsibilities[:,k]) for k in range(K)])
            # print(self.means)

            self.variances = np.array([np.sum([responsibilities[i][k] * (np.dot(np.array(x[i] - self.means[k]).reshape(D, 1), np.array(x[i] - self.means[k]).reshape((1, D)))) for i in range(N)], axis=0) / np.sum(responsibilities[:,k]) for k in range(K)])
            # print(self.variances.shape)

            self.pi_k = np.array([np.sum(responsibilities[:,k])/N for k in range(K)])
            # print(self.pi_k)

            llnew = self.compute_log_likelihood(x)
            # print(t, '  abs(llnew - ll) ', abs(llnew - ll))
            if abs(llnew - ll) < self.e:
                break
            else:
                ll = float(llnew)
        # print('               end ', ll)
        return t

        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE

        D = self.means.shape[1]

        z = np.random.choice(self.n_cluster, N, p=self.pi_k)
        samples = np.zeros((N, D))
        for i, k in enumerate(z):
            mu = self.means[k]
            sigma = self.variances[k]
            samples[i] = np.random.multivariate_normal(mu, sigma)

        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE

        N, D = x.shape

        # print('log likeli', len(means))

        # log_likelihood = 0.0
        # for i in range(N):
        #     pi = 0.0
        #     for k in range(len(means)):
        #         if np.linalg.matrix_rank(variances[k]) != D:
        #             variances[k] = variances[k] + np.identity(D) * 10 ** -3
        #         exp = np.exp(-0.5 * np.dot(np.dot(np.array(x[i]-means[k]).reshape(1,D), np.linalg.inv(variances[k])),np.array(x[i]-means[k]).reshape(D,1)))
        #         pi += pi_k[k] / np.sqrt(((2*np.pi)**D)*np.linalg.det(variances[k])) * exp
        #     log_likelihood += np.log(pi[0][0])
        # log_likelihood = float(log_likelihood)

        joint = np.zeros((N, len(means)))
        for k in range(len(means)):
            mu = means[k]
            if np.linalg.matrix_rank(variances[k]) != D:
                variances[k] = variances[k] + np.identity(D) * 10 ** -3
            sigma = variances[k]
            det = np.linalg.det(sigma)
            denom = np.sqrt((2 * np.pi) ** D * det)
            f = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(sigma)), x - mu), axis=1)) / denom
            joint[:, k] = pi_k[k] * f
        log_likelihood = float(np.sum(np.log(np.sum(joint, axis=1))))

        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE

            D = len(self.mean)
            if np.linalg.matrix_rank(self.variance) != D:
                newVar = self.variance + np.identity(D) * 10 ** -3
                self.c = (2 * np.pi) ** D * np.linalg.det(newVar)
                self.inv = np.linalg.inv(newVar)
            else:
                self.c = (2 * np.pi) ** D * np.linalg.det(self.variance)
                self.inv = np.linalg.inv(self.variance)

            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE

            d = len(x)
            expVal = -0.5 * (np.dot(np.dot(np.array(x - self.mean).reshape(1,d), self.inv), np.array(x - self.mean).reshape(d,1)))
            p = np.exp(expVal) / np.sqrt(self.c)
            p = p[0][0]

            # DONOT MODIFY CODE BELOW THIS LINE
            return p
