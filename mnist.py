
'''
mnist.py is a python program designed to use the MNIST handwritten number
dataset to showcase the basics of neural networks

Works with Python 3.7
'''


import numpy as np
#from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt
import urllib.request as request
import gzip
import pickle
import time


#A bit of setup, to download MNIST
filename = [
    ["training_images","train-images-idx3-ubyte.gz"],
    ["test_images","t10k-images-idx3-ubyte.gz"],
    ["training_labels","train-labels-idx1-ubyte.gz"],
    ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

# if __name__ == '__main__':
#     init()


init()

x, y, _, _ = load()


start = time.time()
x_train = (x[:1000])/255
y_train = y[:1000]

x_val = (x[1000:1200, :])/255
y_val = y[1000:1200]

x_test = (x[1200:1400, :])/255
y_test = y[1200:1400]

#Displaying a random image to visualize the format of the data, each image in X has already been flattened
# im = x[np.random.randint(x.shape[0])].reshape((28, 28))
# plt.imshow(im)
# plt.show()




#Linear Classifier time hehe

class LinearClassifier(object):

    def __init__ (self, features_number, classes_number):
         self.Weights = np.random.randn(features_number, classes_number)
         self.biases = np.zeros((1, classes_number))

    def loss(self, mode, training_features, training_captions, regularization_constant = 0):

        W = self.Weights
        b = self.biases
        X = training_features
        y = training_captions

        N, M = X.shape
        M, C = W.shape

        scores = np.zeros((N, C))
        l = 0
        gradients = {}
        gradients['W'] = np.zeros((M, C))
        gradients['b'] = np.zeros((C, 1))
        #compute scores
        scores = X.dot(W) + b

        if mode == 'test':
            return np.argmax(scores, axis = 1)

        #compute loss using a loss function of your choice
        yi = scores[np.arange(N), y]
        denominator = np.log(np.sum(np.exp((scores)), axis = 1))
        l = np.sum(denominator - yi)/N

        #compute gradients
        #this is the problematic line for Python 2
        d_denom = (1/np.sum(np.exp((scores)), axis = 1, keepdims = True))*np.exp((scores))/N

        dyi = -1/N

        dW = np.transpose(X).dot(d_denom)
        db = np.sum(d_denom, axis = 0)

        yi_onehot = np.eye(10)[y]
        dW += np.transpose(X).dot(yi_onehot)*dyi
        db += np.sum(yi_onehot, axis = 0)*dyi

        gradients['W'] = dW
        gradients['b'] = db

        return scores, l, gradients


    def train(self, X, y, epochs, learning_rate, regularization_constant = 0):
         losses = []
         l = None
         for i in range(epochs):
            scores, l, gradients = self.loss('train', X, y, regularization_constant)
            dW = gradients['W']
            db = gradients['b']
            losses.append(l)
            #update weights
            self.Weights -= learning_rate*dW
            self.biases -= learning_rate*db

         return l, losses


classifier = LinearClassifier(x_train.shape[1],10)


#Complete in the LinearClassifier file the loss function and the training method

scores, loss, gradients = classifier.loss('train', x_train, y_train)

print('Initial loss '+ str(loss)) #initial loss
print('Linear classifier is training')

#tune this parameters and call the function classifier.train(...) to train the network,
#than with classifier.loss(...) try to maximize the validation accuracy tweaking the
#hyperparameter
epochs = 20000
learning_rate = 10e-1
regularization_constant = 1

l, losses = classifier.train(x_train, y_train, epochs, learning_rate, regularization_constant)


end = time.time()
length = end - start
print("Linear classifier  is trained in ", length)

plt.plot(losses)
plt.title("is this loss?")
plt.show()
y_out = classifier.loss('test', x_val, y_val, regularization_constant)



accuracy = np.mean(np.equal(y_out, y_val))

print('Validation accuracy = '+ str(accuracy))


#After you are done tweaking the hyperparameters check your final test accuracy

y_out = classifier.loss('test', x_test, y_test, regularization_constant)

accuracy = np.mean(np.equal(y_out, y_test))

print('test accuracy = '+ str(accuracy))


#Predicting a random image from the dataset
random_data = x[np.random.randint(x.shape[0])]
prediction = classifier.loss('test', np.expand_dims(random_data, axis = 0), y_test, regularization_constant)
print('Prediction = '+ str(prediction))
im = random_data.reshape((28, 28))

plt.imshow(im)
plt.show()

#Predicting another random image from the dataset
random_data2 = x[np.random.randint(x.shape[0])]
prediction = classifier.loss('test', np.expand_dims(random_data2, axis = 0), y_test, regularization_constant)
print('Prediction = '+ str(prediction))
im2 = random_data2.reshape((28, 28))

plt.imshow(im2)
plt.show()

#And again
random_data3 = x[np.random.randint(x.shape[0])]
prediction = classifier.loss('test', np.expand_dims(random_data3, axis = 0), y_test, regularization_constant)
print('Prediction = '+ str(prediction))
im3 = random_data3.reshape((28, 28))

plt.imshow(im3)
plt.show()
