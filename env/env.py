import numpy as np
import random
import yaml
import collections

def showmax(lt):
        index1 = 0                    
        max = 0                         
        for i in range(len(lt)):
            flag = 0                   
            for j in range(i+1,len(lt)): 
                if lt[j] == lt[i]:
                    flag += 1           
            if flag > max:              
                max = flag
                index1 = i
        return lt[index1]    

class timegeo_env(object):
    def showmax(lt):
        index1 = 0                    
        max = 0                         
        for i in range(len(lt)):
            flag = 0                   
            for j in range(i+1,len(lt)): 
                if lt[j] == lt[i]:
                    flag += 1           
            if flag > max:              
                max = flag
                index1 = i
        return lt[index1]           

    def __init__(self, config_path='./config/config.yaml'):
        f = open(config_path)
        self.config = yaml.load(f)
        self.rank_list = np.load(self.config['rank_path'])
        self.track_data = np.loadtxt(self.config['track_path']).astype(int)
        self.pos_num = len(self.rank_list)
        self.count = 0
        self.start_pos = list(self.track_data[:, :-1])
        self.alpha = self.config['alpha']
        # * rank EPR: P(k) ~ k ^ (- \alpha)
        self.traj_length = self.config['traj_length']
        self.max_closest_rank = self.config['max_closest_rank']
        # * normalization factor
        self.normalization = sum([i ** -self.alpha for i in range(1, self.max_closest_rank + 1)])
        self.k_prob = [(x ** (-self.alpha)) / self.normalization for x in range(1, self.max_closest_rank + 1)]

        self.reset()

    def reset(self):
        '''
        self.t = self.count % (self.traj_length - 1)
        self.history_pos = list(self.track_data[self.count // (self.traj_length - 1), :self.t])
        self.current_pos = self.start_pos[self.count // (self.traj_length - 1)][self.t]
        '''
        self.t = 0
        if self.count == 3000:
            self.count = 0
        self.history_pos = list(self.track_data[self.count, :1])
        self.current_pos = self.start_pos[self.count][0]
        home_point = list(self.track_data[self.count, 0:46])
        self.home_point = showmax(home_point)
        pre_pos_count = len(list(set(self.history_pos)))
        stay_time = 0
        for pos in reversed(self.history_pos):
            if pos == self.current_pos:
                stay_time += 1
            else:
                break
        return self.current_pos, self.t, self.history_pos, self.home_point, pre_pos_count, stay_time

    def step(self, action, history_pos, home_point, eval=False):
        self.t += 1
        done = (self.t >= self.traj_length - 1)
        self.home_point = home_point
        self.history_pos = history_pos
        self.history_pos.append(self.current_pos)

        if not eval:
            if done:
                if self.count % (np.prod(self.track_data.shape) - self.track_data.shape[0] - 1) == 0 and self.count != 0:
                    self.count = -1
                else:
                    pass
                self.count += 1
        
        if action == 0:
            # * action == 0: stay
            pass
        elif action == 1:
            # * action == 1: explore
            rp = random.uniform(0, 1)
            prob_accum = 0
            for i, p_k in enumerate(self.k_prob):
                prob_accum += p_k
                if prob_accum >= rp:
                    self.current_pos = self.rank_list[int(self.current_pos), i]
                    break
        elif action == 2:
            # * action == 2: home
            self.current_pos = self.home_point
            if len(self.history_pos)==47 and len(set(self.history_pos))==1:
                self.current_pos = self.rank_list[int(self.current_pos), 1]

        elif action == 3:
            # * action == 3: return

                his=[]
                for i in range(len(self.history_pos)):
                     if self.history_pos[i] != self.home_point:
                         his.append(self.history_pos[i])
                if len(his) == 0:
                     pass
                else:
                     pos_prob = collections.Counter(his)
                     pos = list(pos_prob.keys())
            
                     prob = pos_prob.values()
                     prob = [p / len(his) for p in prob]
                     pref_return = np.random.choice(pos, p=prob)
                     self.current_pos = pref_return

        pre_pos_count = len(list(set(self.history_pos)))
        #print(self.history_pos)
        stay_time = 0
        for pos in reversed(self.history_pos):
            if pos == self.current_pos:
                stay_time += 1
            else:
                break

        return self.current_pos, self.t, done, self.history_pos, self.home_point, pre_pos_count, stay_time

    
    def set_state(self, pos=None, t=None):
        self.current_pos = pos if pos is not None else self.current_pos
        self.t = t if t is not None else self.t

        
#env = timegeo_env()

