import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SVM:
    def __init__(self,data) -> None:
        # Data is of the form x where is a vecrtor or a tensor of n dimensions for simpliocity as defined in main
        self.data = data
    
    def fit(self):
        all_data = []
        for cls in self.data:
            for features in self.data[cls]:
                for feature in features:
                    all_data.append(feature)
        
        self.min_feature_value = np.min(all_data)
        self.max_feature_value = np.max(all_data)
        opt_dict = {}
        all_data = None
        # For w
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]
        transforms = [[1,1],
                    [-1,1],
                    [-1,-1],
                    [1,-1]]
        # Width  = 2/||w||
        # Hyperplane, X.W + b = 0
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                    self.max_feature_value*b_range_multiple,
                                    step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            

    def predict(self, feature):
        classification = np.sign(np.dot(self.w,np.array(feature)) + self.b)
        return classification

if(__name__ == "__main__"):
    data = {
        -1:np.array([[2,3],[3,3],[1,3],[2,1],[4,3],[3,3],[2,1],[2,4],[1,1],[2,2],[3,1],[4,2]]),
        1:np.array([[7,8],[9,8],[7,9],[8,9],[7,7],[8,7],[8,8],[9,9],[9,8],[6,7],[6,8],[6,9]])
    }

    svm = SVM(data)
    svm.fit()
    print(svm.w,svm.b)
    #[[plt.scatter(i[0],i[1], color =j) for i in data[j]]for j in data]
    [[plt.scatter(i[0],i[1]) for i in data[j]]for j in data]
    x = np.arange(0,10)
    plt.plot(x*svm.w+svm.b)
    plt.show()