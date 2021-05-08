import pandas as pd
import numpy as np

class K_means(object):
    def __init__(self, k=5, threshold=0.00001, max_iteration=30):
        self._k = k
        self._threshold = threshold
        self._max_iteration = max_iteration
        self.labels = {}

    def choose_centers(self):
        first_id = np.random.randint(0, self.data_length)
        self._centers[0] = data[first_id]
        dist_note = np.zeros(self.data_length)
        dist_note += 1000000000
        for i in range(1, self._k):
            for j in range(self.data_length):
                dist = np.linalg.norm(data[j] - self._centers[i - 1])
                if dist < dist_note[j]:
                    dist_note[j] = dist
            next_id = dist_note.argmax()
            self._centers[i] = data[next_id]

    def clustering(self, data):
        self.data_length = len(data)
        self._centers = {}
        '''
        random_centers = np.random.randint(0, self.data_length, size=self._k)
        for i in range(self._k):
            self._centers[i] = data[random_centers[i]]#这里可以优化
        '''
        self.choose_centers()

        for _ in range(self._max_iteration):
            self._classification = {}
            for j in range(self._k):
                self._classification[j] = []
            for j in range(self.data_length):
                distances = []
                for center in self._centers:
                    distances.append(np.linalg.norm(data[j]-self._centers[center]))
                label = distances.index(min(distances))
                self._classification[label].append(data[j])
                self.labels[j] = label
            previous_centers = self._centers.copy()

            for i in self._classification:
                self._centers[i] = np.average(self._classification[i], axis=0)

            is_continue = False
            for center in self._centers:
                centers_old = previous_centers[center]
                centers_new = self._centers[center]
                if np.sum(np.abs((centers_new-centers_old)/centers_old*100)) > self._threshold:
                    is_continue = True
            if is_continue == False:
                break

    def predict(self):
        radius = []
        for i in range(self._k):
            radius.append(np.max(np.linalg.norm(self._centers[i]-self._classification[i], axis=1)))
        radius = np.array(radius)
        self.order = np.argsort(radius)
        self.results = np.zeros(self.data_length)
        for i in range(self.data_length):
            self.results[i] = int(np.argwhere(self.order == self.labels[i]))
        self.results = self.results.astype(np.int32)

    def output(self, STORE_PATH):
        frame = {"category": self.results}
        df = pd.DataFrame(frame)
        df.index.name = 'id'
        df.to_csv(STORE_PATH)


DATA_PATH = 'course1.csv'
STORE_PATH = "results.csv"
data = pd.read_csv(DATA_PATH, index_col='PID')
data = data.values
k_means = K_means(k=5)
k_means.clustering(data)
k_means.predict()
k_means.output(STORE_PATH)


