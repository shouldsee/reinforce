import gym
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential,load_model
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    if 1:
        for i in range(0, len(l), n):
    #         print(l.shape)
    #         print(l[i:i + n,:].shape)
    #         print('yield l[%d:%d + %d,:]'%(i,i,n))
            yield l[i:i + n]

class PGAgent:
    def __init__(self, state_size, action_size,AgentName = 'tst'):
#         self.AgentName = 'pong_minimal-s5L1b-tst'
        self.name = AgentName
        self.AgentFile = 'Models/%s.h5'%AgentName;
        self.LogName = 'Models/%s.log'%AgentName;
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.summary = self.model.summary;

    def _build_model(self):
        model = Sequential()
        model.add(Reshape((80, 80, 1), input_shape=(self.state_size,)))
        model.add(Convolution2D(25, (6, 6), subsample=(3, 3), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Convolution2D(5, (6, 6), subsample=(1, 1), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Convolution2D(5, (6, 6), subsample=(1, 1), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Convolution2D(5, (6, 6), subsample=(1, 1), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Flatten())
#         model.add(Dense(20, activation='relu', init='he_uniform'))
#         model.add(Dense(20, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        # state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self,rewards,batch_size = 2000,verbose = 0):
        gradients = np.vstack(self.gradients)
        # rewards = np.vstack(self.rewards)
        # rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards,keepdims=1)) / np.std(rewards)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
#         print(X.shape)
#         self.model.train_on_batch(X, Y)
#         X = np.expand_dims(X,axis = 1)
#         Y = np.expand_dims(Y,axis = 1)
    
#         gen = ((X[i],Y[i]) for i in range(len(X)));
#         zipped = zip(X,Y)
        Xgen = chunks(X,batch_size);
        Ygen = chunks(Y,batch_size);
        gen = (x for x in zip(Xgen,Ygen))
#         gen1 = ((x,y) for x,y in zip(chunks(X,batch_size).next(),chunks(Y,batch_size).next()))
#         print(gen1.next()[1].shape)
#         print(chunks(X,batch_size).next().shape)
#         print()
        bmax = max(X.shape[0]//batch_size,1)
#         print(gen)
#         print(bmax)
        try:
            self.model.fit_generator(gen, steps_per_epoch = bmax, epochs = 1,verbose = verbose, max_q_size=1)
        except StopIteration:
            pass
#         self.model.fit(X,Y,  epochs = 1,verbose = verbose)
    
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, fname=None):
        if not fname:
            fname = self.AgentFile;
        global episode
        self.model = load_model(fname);
        self.summary = self.model.summary;
        
    def readlog(self, LogName=None):
        if not LogName:
            LogName = self.LogName;
        with open(LogName,'rb') as f:
                first = f.readline()      # Read the first line.
                if not first.rstrip('\n'):
#                     print('nothing!')
                    self.episode = 0;
                else:
                    f.seek(-2, 2)             # Jump to the second last byte.
                    while f.read(1) != b"\n": # Until EOL is found...
                        f.seek(-2, 1)         # ...jump back the read byte plus one more.
                    last = f.readline() 
                    lst = last.split('\t');
                    eind = lst.index('Episode')+1;
                    self.episode = int(lst[eind]);

    def save(self, fname=None):
        if not fname:
            fname = self.AgentFile;
        self.model.save(fname);
    def writelog(self, msg, LogName = None):
        if not LogName:
            LogName = self.LogName
        with open(LogName,'a+') as LogFile:
            LogFile.write(msg+'\n');
    def newlog(self):
        open(self.LogName,'w').close();



def main(agent, verbose = 1):
    if __name__ == "__main__":
        env = gym.make("Pong-v0")
        state = env.reset()
        prev_x = None
        score = 0
        episode = 0
    if resume:
        agent.load();
        agent.readlog();
        episode = agent.episode;
    while episode < agent.episode + nb_episode:
        if render:
                env.render()

        cur_x = preprocess(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        prev_x = cur_x

        action, prob = agent.act(np.expand_dims(x,1).T)
        state, reward, done, info = env.step(action)
        score += reward
        agent.remember(x, action, prob, reward)

        if done:
            episode += 1
            if render: break;
            rewards = np.vstack(agent.rewards)
            rewards = agent.discount_rewards(rewards)
            grads = np.vstack(agent.gradients);
    #             pca.fit (grads )
    #             lvar = pca.explained_variance_ratio_[0];
            lvar = 0;
            var_lst = np.var(grads,axis = 0);
            L1_lst = abs(np.mean(grads,axis = 0));
            L1_norm = np.mean(L1_lst)
            all_var = np.mean(var_lst);            

            agent.train(rewards,batch_size = batch_size,verbose = max(verbose-1,0))
            msg = '%s\t%d\t%s\t%f\t%s\t%f\t%s\t%f\t%s\t%f' % ('Episode',episode,'Score', score,'largest_variance',lvar,'all_var',all_var,'L1_norm',L1_norm);
            if verbose:
                print(msg);
            agent.writelog(msg)

            score = 0
            state = env.reset()
            prev_x = None
            if episode > 1 and episode % 10 == 0 and not render:
                agent.save()
    print('Agent: %s finshed trainning episode %d to %d'%(agent.name, agent.episode,episode))
            
def preprocess(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()



# agent.summary();