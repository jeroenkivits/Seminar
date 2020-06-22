import numpy as np
from scipy import optimize

"""
SVM class that contains all function needed to train a SVM classifier, 
private and non-private.
"""
class SVM:
    def __init__(self, private, labda, h):
        self.private = private
        self.labda = labda
        self.h = h

    """"
    Function that trains the SVM classifier.
    """
    def fit(self, training_data, epsilon_p):
        # Split dataframe between labels and d
        df_train_y = training_data["dependent"]
        df_train_x = training_data.drop(columns="dependent")

        train_x = df_train_x.values
        train_y = df_train_y.values

        self.n = train_x.shape[0]
        self.features = train_x.shape[1]

        # Create initial guess for optimization
        f0 = np.ones(self.features)

        if self.private:
            epsilon_p_prime = epsilon_p-np.log(1+((1/self.h)/(self.n*self.labda))+(
                        ((1/(2*self.h))**2)/((self.n**2)*(self.labda**2))))

            # check which delta should be used
            if epsilon_p_prime > 0:
                delta = 0
            else:
                delta = ((1/(2*self.h))/(self.n*(np.exp(epsilon_p/4)-1)))-self.labda
                epsilon_p_prime = epsilon_p/2

            # create noise according to epsilon_p_prime
            noise = self.noise_diffpriv(self.features, epsilon_p_prime)

            # optimize function func with BFGS method
            sol = optimize.fmin_bfgs(self.func, f0, args=(train_x, train_y, noise), disp=False)
            f = sol

            self.f = f + (delta*np.linalg.norm(f)**2)/2

        else:
            # create vector with zeros to pass as noise
            noise = np.zeros(self.features)

            # optimize function func with BFGS method
            sol = optimize.fmin_bfgs(self.func, f0, args=(train_x, train_y, noise), disp=False)
            self.f = sol


    """"
    Callback function available to keep tabs on the optimization process,
    only necessary to add callback=call to the scipy optimize function.
    optimize.fmin_bfgs(self.func, f0, args=(train_x, train_y, noise), disp=False, callback=call)
    """
    def call(self, data):
        print('[Iteration finished]')

    """"
    This function calculates regularized emperical loss according to 
    """
    def func(self, f, train_x, train_y, noise):
        # calculate position of training data to predictor f
        z = np.dot(np.transpose(f), np.transpose(train_x)) * train_y

        conditions = [
            z > 1 + self.h,
            np.abs(1 - z) <= self.h,
            z < 1 - self.h
        ]

        choices = [
            0,
            (1 / 4 * self.h) * (1 + self.h - z),
            1 - z
        ]

        loss = np.select(conditions, choices)

        # return regularized emperical loss, noise is added in this step
        # noise set to zero will not be added to the regularized emperical loss, since it renders the 1/n * Transpose(b) . f zero
        return np.mean(loss) + (self.labda / 2 * (np.linalg.norm(f) ** 2)) + ((1 / self.n) * np.dot(np.transpose(noise), f))

    """
    This function predicts the position of the test data relative to the predictor vector f
    and returns the error rate.    
    """
    def evaluate(self, test_data):
        df_test_y = test_data["dependent"]
        df_test_x = test_data.drop(columns="dependent")

        test_x = df_test_x.values
        test_y = df_test_y.values

        errors = 0
        for idx, datapoint in enumerate(test_x):
            pred = np.sign(np.dot(np.transpose(self.f), datapoint))

            if pred != test_y[idx]:
                errors += 1

        return errors/test_x.shape[0]

    """"
    Construct noise with exponential distribution with beta = 2/epsilon_p_prime
    """
    def noise_diffpriv(self, dim, epsilon_p_prime):
        b = np.random.exponential(2/epsilon_p_prime, dim)
        return b