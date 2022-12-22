from tensorflow.keras.layers import Flatten, Embedding, Input, Concatenate, Dense, Add, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import L2


class deep_residual_network_recommender:
    def __init__(self, K):
        self.K = K  # latent dim
        self.mu = None
        self.predict_model = None

    def fit(self, ratings_df, ratings_df_test, users_col_name, items_col_name, ratings_col_name, n_epochs=25, learing_rate=0.08, reg=0.):
        self.mu = ratings_df[ratings_col_name].mean()  # ratings global bias

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

        # main branch
        u_i_correlation = Dot(axes=1)([users_emb, items_emb])  # output shape=(N, 1)
        x_main = Add()([u_i_correlation, users_bias, items_bias])  # output shape=(N, 1)

        # residuals branch
        u_i_concat = Concatenate()([users_emb, items_emb])  # output shape=(N, K + K)
        x_res1 = Dense(400, activation='relu')(u_i_concat)  # output shape=(N, 400)
        x_res1 = Dense(1)(x_res1)  # output shape=(N, 1)

        x = Add()([x_main, x_res1])

        model = Model(inputs=[user_input, item_input], outputs=x)
        model.compile(loss='mse', optimizer=SGD(learning_rate=learing_rate, momentum=0.9), metrics=['mse'])
        print(model.summary())

        self.predict_model = Model(inputs=[user_input, item_input], outputs=x)

        x = [ratings_df[users_col_name].values, ratings_df[items_col_name].values]
        y = ratings_df[ratings_col_name].values - self.mu
        x_test = [ratings_df_test[users_col_name].values, ratings_df_test[items_col_name].values]
        y_test = ratings_df_test[ratings_col_name].values - self.mu

        history = model.fit(x, y, epochs=n_epochs, batch_size=128, validation_data=(x_test, y_test))
        return history

    def predict(self, users, items):
        r_hat = self.predict_model.predict([users, items]) + self.mu
        return r_hat
