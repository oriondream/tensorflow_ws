import numpy as np

class KFold:

    def __init__(self, data = None, k = None):
        if (data is not None) and (k is not None):
            self.data = data
            self.k = k
            self.slice_size = int( list(data.data.shape)[0] / k )

    def get_validation(self, fold):
        return self.data [fold * self.slice_size : (fold+1) * self.slice_size]

    def get_training(self, fold):
        return np.delete(self.data, slice(fold * self.slice_size,(fold+1) * self.slice_size))

    def test(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        kf = KFold(arr,3)

        k = 1

        B = kf.get_validation(k)
        print(B)

        A = kf.get_training(k)
        print(A)

KFold().test()