import numpy as np

class KalmanFilter():
    def __init__(self):
        self.dt = 1
        self.A = np.matrix([[1,0],
                    [0,1]])
        self.B = np.matrix([[1,0],
                    [0,1]])
        self.u = np.matrix([[1],
                    [1]])
        self.Q = np.matrix([[20,0],
                    [0,20]])

        self.H = np.matrix([[1,0],
                    [0,1]]) 
        self.R = np.matrix([[50,0],
                    [0,50]])

        self.x = np.matrix([[320],
                    [180]])
        self.P = np.matrix([[1,0],
                    [0,1]])
        self.I = np.eye(2)

    def predict(self):
        # Prediction step
        self.x = self.A*self.x + self.B*self.u
        self.P = self.A*self.P*self.A.T + self.Q
        return self.x

    def correct(self, z):
        # Correction/Update step
        K = self.P*self.H.T*np.linalg.inv(self.H*self.P*self.H.T + self.R)
        self.x = self.x + K*(z - self.H*self.x)
        self.P = (self.I - K*self.H)*self.P
        return self.x