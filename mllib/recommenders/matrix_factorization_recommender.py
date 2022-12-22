import numpy as np

from tensorflow.keras.layers import Flatten, Embedding, Input, Dot, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import L2



class matrix_factorization_recommender:
    def __init__(self, K):
        self.K = K  # latent dim
        self.W = None  # users representation
        self.b = None  # user representation bias
        self.U = None  # items representation
        self.c = None  # item representation bias
        self.mu = None  # ratings global bias

    def fit_by_als(self, user2item, item2user, user_item2rating, user_item2rating_test, n_epochs=25, reg=20., initial_weights=None):
        # fit weights by alternating least squares
        self.mu = np.mean(list(user_item2rating.values()))  # ratings global bias

        N = np.max(list(user2item.keys())) + 1
        M = np.max(list(item2user.keys())) + 1

        if initial_weights is None:
            self.W = np.random.randn(N, self.K)  # users representation
            self.b = np.zeros(N)  # user representation bias
            self.U = np.random.randn(M, self.K)  # items representation
            self.c = np.zeros(M)  # item representation bias
        else:
            self.W, self.b, self.U, self.c = initial_weights

        user2ratings = {user: [user_item2rating[(user, j)] for j in items] for user, items in user2item.items()}
        item2ratings = {item: [user_item2rating[(i, item)] for i in users] for item, users in item2user.items()}
        #item2ratings_test = {item: [user_item2rating_test[(i, item)] for i in users] for (item, user), rating in user_item2rating_test.items()}

        history = {'train_cost': [],
                   'test_cost': []}
        for epoch in range(n_epochs):
            cost, _ = self._calc_cost(user_item2rating)
            test_cost, _ = self._calc_cost(user_item2rating_test)
            history['train_cost'].append(cost)
            history['test_cost'].append(test_cost)
            print(f'epoch: {epoch}, train cost: {cost}, test cost: {test_cost}')

            for i in range(N):
                items_i = user2item[i]
                U_i = self.U[items_i]

                # update self.W
                aW = U_i.T.dot(U_i) + reg * np.eye(self.K)
                r = np.array(user2ratings[i])
                bW = (r - self.b[i] - self.c[items_i] - self.mu).dot(U_i)

                # update self.b
                ab = len(items_i) + reg
                bb = np.sum(r - self.U[items_i].dot(self.W[i]) - self.c[items_i] - self.mu)

                self.W[i] = np.linalg.solve(aW, bW)
                self.b[i] = bb / ab

            for j in range(M):
                users_j = item2user[j]
                W_j = self.W[users_j]

                # update self.U
                aU = W_j.T.dot(W_j) + reg * np.eye(self.K)
                r = np.array(item2ratings[j])
                bU = (r - self.b[users_j] - self.c[j] - self.mu).dot(W_j)

                # update self.c
                ac = len(users_j) + reg
                bc = np.sum(r - self.W[users_j].dot(self.U[j]) - self.b[users_j] - self.mu)

                self.U[j] = np.linalg.solve(aU, bU)
                self.c[j] = bc / ac
        return history

    def fit_by_tf2(self, ratings_df, ratings_df_test, users_col_name, items_col_name, ratings_col_name, n_epochs=25, learing_rate=0.08, reg=0.):
        # fit weights with tenforflow 2
        self.mu = ratings_df[ratings_col_name].mean()  # ratings global bias
        print(self.mu)

        N = ratings_df[users_col_name].max() + 1
        M = ratings_df[items_col_name].max() + 1

        user_input = Input(shape=(1, ))  # output shape=(N, 1)
        item_input = Input(shape=(1, ))  # output shape=(N, 1)

        users_emb_layer = Embedding(input_dim=N, output_dim=self.K, embeddings_regularizer=L2(reg))
        items_emb_layer = Embedding(input_dim=M, output_dim=self.K, embeddings_regularizer=L2(reg))
        users_bias_layer = Embedding(input_dim=N, output_dim=1, embeddings_regularizer=L2(reg))
        items_bias_layer = Embedding(input_dim=M, output_dim=1, embeddings_regularizer=L2(reg))

        users_emb = users_emb_layer(user_input)  # output shape=(N, 1, K)
        items_emb = items_emb_layer(item_input)  # output shape=(N, 1, K)
        users_bias = users_bias_layer(user_input)  # output shape=(N, 1, 1)
        items_bias = items_bias_layer(item_input)  # output shape=(N, 1, 1)

        users_emb = Flatten()(users_emb)  # output shape=(N, K)
        items_emb = Flatten()(items_emb)  # output shape=(N, K)
        users_bias = Flatten()(users_bias)  # output shape=(N, 1)
        items_bias = Flatten()(items_bias)  # output shape=(N, 1)

        u_i_correlation = Dot(axes=1)([users_emb, items_emb])  # output shape=(N, 1)
        x = Add()([u_i_correlation, users_bias, items_bias])  # output shape=(N, 1)

        model = Model(inputs=[user_input, item_input], outputs=x)
        model.compile(loss='mse', optimizer=SGD(learning_rate=learing_rate, momentum=0.9), metrics=['mse'])
        print(model.summary())

        x = [ratings_df[users_col_name].values, ratings_df[items_col_name].values]
        y = ratings_df[ratings_col_name].values - self.mu
        x_test = [ratings_df_test[users_col_name].values, ratings_df_test[items_col_name].values]
        y_test = ratings_df_test[ratings_col_name].values - self.mu

        history = model.fit(x, y, epochs=n_epochs, batch_size=128, validation_data=(x_test, y_test))

        self.W = users_emb_layer.embeddings.numpy()
        self.b = users_bias_layer.embeddings.numpy().flatten()
        self.U = items_emb_layer.embeddings.numpy()
        self.c = items_bias_layer.embeddings.numpy().flatten()
        return history

    def predict(self, user, item):
        r_hat = self.W[user].dot(self.U[item]) + self.b[user] + self.c[item] + self.mu
        return r_hat

    def _calc_cost(self, user_item2rating):
        predictions = []
        targets = []
        for (user, item), target in user_item2rating.items():
            predictions.append(self.predict(user, item))
            targets.append(target)

        errors = np.array(predictions) - np.array(targets)
        #reg = reg_scale * (np.ling.norm(self.W) + np.ling.norm(self.b) + np.ling.norm(self.U) + np.ling.norm(self.c))
        cost = errors.dot(errors) / len(errors)  # + reg
        return cost, errors
