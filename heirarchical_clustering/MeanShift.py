import numpy as np

class MeanShift:
    def __init__(self,data, radius=4):
        self.r = radius
        self.data = data
    
    def fit(self):
        X = self.data
        centroids = {}
        
        for i in range(len(X)):
            centroids[i] = X[i]
        
        while True:
            newCentroids = []
            
            for i in centroids:
                centroid = centroids[i]
                in_band = []
                
                for j in X:
                    if np.linalg.norm (centroid-j) <= self.r:
                        in_band.append(j)
                
                new_centroid = np.average(in_band, axis=0)
                newCentroids.append(tuple(new_centroid))
                
            uniques = sorted(list(set(newCentroids)))
            prev_centroids = dict(centroids)
            centroids = {}
            for i in uniques:
                centroids[i] = np.array(uniques[i])
            
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            
            if optimized:
                break
    
        self.centroids = centroids
    
    def predict(self,data):
        #compare distance to either centroid
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification


class MeanShift:
    def __init__(self,data, radius=None, wieght_mag = 100):
        self.r = radius
        self.data = data
        self.wmag = wieght_mag
        
        self.weights = [i for i in range(self.wmag)][::-1]
        
        if(self.r == None):
            all_avg_centroid = np.average(data,axis=0)
            all_data_mag = np.linalg.norm(all_avg_centroid)
            self.r = all_data_mag/self.wmag
    
    def fit(self):
        X = self.data
        centroids = {}
        
        for i in range(len(X)):
            centroids[i] = X[i]
        
        while True:
            newCentroids = []
            
            for i in centroids:
                centroid = centroids[i]
                in_band = []
                
                for j in X:
                    #if np.linalg.norm(featureset-centroid) < self.radius:
                    #    in_bandwidth.append(featureset)
                    distance = np.linalg.norm(j-centroid)
                    if distance == 0:
                        distance = 0.00000000001
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1

                    to_add = (self.weights[weight_index]**2)*[j]
                    in_band +=to_add
                
                new_centroid = np.average(in_band, axis=0)
                newCentroids.append(tuple(new_centroid))
                
            uniques = sorted(list(set(newCentroids)))
            to_pop = []
            for i in uniques:
                for ii in [i for i in uniques]:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        #print(np.array(i), np.array(ii))
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)
            centroids = {}
            for i in uniques:
                centroids[i] = np.array(uniques[i])
            
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            
            if optimized:
                break
    
        self.centroids = centroids
    
    def predict(self,data):
        #compare distance to either centroid
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification
