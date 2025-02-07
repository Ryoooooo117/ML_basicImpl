import json
import numpy as np


###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension weight vector
    - lamb: lambda used in pegasos algorithm

    Return:
    - obj_value: the value of objective function in SVM primal formulation
    """
    # you need to fill in your solution here

    obj_value = 0.5 * lamb * (np.linalg.norm(w) ** 2) + np.mean(np.maximum(0, 1 - y * np.dot(X,w.T).T))

    return obj_value


###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the total number of iterations to update parameters

    Returns:
    - learnt w
    - train_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    train_obj = []


    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch

        # you need to fill in your solution here

        A_tp = [i for i in A_t if (ytrain * np.dot(Xtrain,w).T).T[i] < 1]
        eta = 1.0 / (lamb * iter)
        wDelta = np.sum(((ytrain * Xtrain.T).T)[A_tp], axis=0)                      # wDelta dim: [D]
        w_t_half = (1.0 - eta * lamb) * w.reshape(D) + eta / k * wDelta             # w dim: D x 1, w_t_half dim: [D]
        w_t = min(1, 1 / np.sqrt(lamb) / np.linalg.norm(w_t_half)) * w_t_half
        loss = objective_function(Xtrain, ytrain, w_t, lamb)
        # if iter % 10 == 1:
        #     print("iter=", iter, " selected num=", len(A_tp), A_t.shape, " eta=", eta, " loss=", loss)
        w = w_t
        train_obj.append(loss)

    return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w_l):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
 
    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here

    pred = np.dot(Xtest,w_l)
    yPred = np.where(pred > 0, 1, -1)
    # print('y_pred ', y_pred)
    # ta = np.mean((ytest == y_pred).astype(int))
    test_acc = 0
    for i in range(len(ytest)):
        if ytest[i] == yPred[i]:
            test_acc += 1
    test_acc = test_acc / len(ytest)
    return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    # max_iterations = 20
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    # test_acc, train_obj = pegasos_mnist() # results on mnist
    # print('mnist test acc \n')
    # for key, value in test_acc.items():
    #     print('%s: test acc = %.4f \n' % (key, value))
    #
    # with open('pegasos.json', 'w') as f_json:
    #     json.dump([test_acc, train_obj], f_json)
    line = "00000000"
    print(process(line))

def process(line):

    if len(line) != 8:
        return "INVALID"

    firstSix = line[:6]
    lastTwo = line[-2:].lower()
    try:
        decimalStr = str(int(firstSix,16))
    except:
        return "INVALID"

    sum = 0
    for num in decimalStr:
        sum += int(num)
    print(firstSix, ' ', lastTwo, ' ', decimalStr, ' ', hex(sum))
    if hex(sum)[2:] == lastTwo:
        return "VALID"
    return "INVALID"

if __name__ == "__main__":
    main()
