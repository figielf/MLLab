import numpy as np


class q_learning_agent(object):
    def __init__(self, state_size, action_size, model_factory):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = model_factory()

    def act(self, state):
        p = np.random.rand()
        if p < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            q_hat = self.model.predict(state)
            action = np.argmax(q_hat[0])
        return action

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            new_state_q_hat = self.model.predict(next_state)
            target = reward + self.gamma * np.amax(new_state_q_hat, axis=1)

        # extend target on every other action so that mse for other actions are always 0
        target_all_actions = self.model.predict(state)
        target_all_actions[:, action] = target

        # run one training step
        mse = self.model.fit_one_batch(state, target_all_actions)

        #self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return mse

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)
