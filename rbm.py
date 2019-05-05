import numpy as np
import pandas as pd

VISIBLE_SIZE = 400
HIDDEN_SIZE = 90

EPOCH = 1000
LR = 0.1

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def getData(csv_file):
    raw_data = pd.read_csv(csv_file)
    data = raw_data.iloc[:,12:].values.astype(np.float32)
    label = raw_data.iloc[:,2].values
    return data, label

class RBM(object):
    def __init__(self, train_set, test_set):
        super(RBM, self).__init__()
        self.visible_size = VISIBLE_SIZE
        self.hidden_size = HIDDEN_SIZE

        self.train_data, self.train_label = train_set
        self.test_data, self.test_label = test_set

        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
                        low=-0.1 * np.sqrt(600. / (self.hidden_size + self.visible_size)),
                       	high=0.1 * np.sqrt(600. / (self.hidden_size + self.visible_size)),
                       	size=(self.visible_size, self.hidden_size)))
        
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)
    
    def getProbs(self, data, t=0):
        if t == 0:
            activations = np.dot(data, self.weights)
        else:
            activations = np.dot(data, self.weights.T)
        probs = sigmoid(activations)
        probs[:,0] = 1
        return probs

    def solve(self):
        set_size = self.train_data.shape[0]
        self.train_data = np.insert(self.train_data, 0, 1, axis=1)

        for epoch in range(EPOCH):
            pos_hidden_probs = self.getProbs(self.train_data)
            pos_hidden_states = pos_hidden_probs > np.random.rand(set_size, self.hidden_size + 1)
            pos_associations = np.dot(self.train_data.T, pos_hidden_probs)
            neg_visible_probs = self.getProbs(pos_hidden_states, 1)
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = sigmoid(neg_hidden_activations)
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            self.weights += LR * ((pos_associations - neg_associations) / set_size)

            if (epoch % 100 == 99):
                loss = np.sum((self.train_data - neg_visible_probs) ** 2) / set_size
                print('Epoch: %5d\t#Loss: %.3f' % (epoch + 1, loss))
    
    def evaluate(self):
        pass

if __name__ == '__main__':
    # train_set = getData('./data/train.csv')
    train_set = getData('./data/test.csv')
    test_set = getData('./data/test.csv')
    rbm = RBM(train_set, test_set)
    rbm.solve()