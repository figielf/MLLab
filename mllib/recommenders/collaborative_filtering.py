import numpy as np

from sortedcontainers import SortedList


class collaborative_filtering:
    def __init__(self, k_neighbours=25, min_common_items=5):
        self.k_neighbours = k_neighbours
        self.min_common_items = min_common_items
        self.user_biases = None
        self.neighbours = None
        self.user_item_ratings = None

    def fit(self, user2item, item2user, user_item2rating):
        self.user_item_ratings = user_item2rating

        assert len(user2item) == len(set(user2item.keys()))
        assert len(item2user) == len(set(item2user.keys()))
        assert len(user_item2rating) == len(set(user_item2rating.keys()))
        nU = np.max(list(user2item.keys())) + 1


        self.user_biases = np.zeros(nU)
        user_deviation_stds = np.zeros(nU)
        for user in range(nU):
            items = user2item[user]
            user_ratings = np.array([self.user_item_ratings[(user, item)] for item in items])
            b = np.mean(user_ratings)
            self.user_biases[user] = b
            user_deviations = user_ratings - b
            user_deviation_stds[user] = np.sqrt(user_deviations.dot(user_deviations))

        uu_common_items = {}
        self.neighbours = []
        for user_i in range(nU):
            items_i = user2item[user_i]
            top_weights = SortedList()
            for user_j in range(nU):
                items_j = user2item[user_j]
                if user_i != user_j:
                    common_items = set(items_i).intersection(items_j)
                    uu_common_items[(user_i, user_j)] = common_items
                    if len(common_items) > self.min_common_items:
                        ui1 = np.array([self.user_item_ratings[(user_i, item)] for item in common_items]) - self.user_biases[user_i]
                        ui2 = np.array([self.user_item_ratings[(user_j, item)] for item in common_items]) - self.user_biases[user_j]

                        weight = ui1.dot(ui2) / (user_deviation_stds[user_i] * user_deviation_stds[user_j])
                        top_weights.add((-weight, user_j))
                        if len(top_weights) > self.k_neighbours:
                            top_weights.pop(index=-1)
            self.neighbours.append(top_weights)

        return 0

    def predict(self, user, item):
        neighbours_weighted_deviations = []
        neighbours_abs_weights = []
        for neg_w_j, user_j in self.neighbours[user]:
            if (user_j, item) in self.user_item_ratings:
                user_j_deviation = self.user_item_ratings[(user_j, item)] - self.user_biases[user_j]
                neighbours_weighted_deviations.append(-neg_w_j * user_j_deviation)
                neighbours_abs_weights.append(abs(neg_w_j))

        if len(neighbours_weighted_deviations) > 0:
            pred_deviation = np.sum(neighbours_weighted_deviations) / np.sum(neighbours_abs_weights)
        else:
            pred_deviation = 0

        pred_rating = pred_deviation + self.user_biases[user]

        pred_rating = max(pred_rating, 0.5)
        pred_rating = min(pred_rating, 5)

        return pred_rating
