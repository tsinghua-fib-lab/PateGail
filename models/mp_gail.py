
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import yaml
import random
import os
import setproctitle
import time as tm
from torch.multiprocessing import Process, Manager,Pool
import torch.multiprocessing as mp
from tqdm import tqdm
from models.replay_buffer import replay_buffer
from models.net import ATNetwork, Discriminator
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class gail(object):
    def __init__(self, env, file, seed, beta, noise, config_path='./config/config.yaml', eval=False):
        f = open(config_path)
        assert type(file) == dict
        self.config = yaml.load(f)
        self.env = env
        self.episode = self.config['episode']
        self.capacity = self.config['capacity']
        self.gamma = self.config['gamma']
        self.lam = self.config['lam']
        self.value_learning_rate = self.config['value_learning_rate']
        self.policy_learning_rate = self.config['policy_learning_rate']
        self.discriminator_learning_rate = self.config['discriminator_learning_rate']
        self.batch_size = self.config['batch_size']
        self.policy_iter = self.config['policy_iter']
        self.disc_iter = self.config['disc_iter']
        self.value_iter = self.config['value_iter']
        self.epsilon = self.config['epsilon']
        self.entropy_weight = self.config['entropy_weight']
        self.train_iter = self.config['train_iter']
        self.clip_grad = self.config['clip_grad']
        self.file = file
        self.alpha = 10
        self.beta = beta
        self.noise = noise
        self.action_dim = 4
        self.total_locations = self.config['total_locations']
        self.time_scale = self.config['time_scale']
        self.loc_embedding_dim = self.config['loc_embedding_dim']
        self.tim_embedding_dim = self.config['tim_embedding_dim']
        self.embedding_net = self.config['embedding_net']
        self.embedding_dim = self.loc_embedding_dim + self.tim_embedding_dim
        self.hidden_dim = self.config['hidden_dim']
        self.bidirectional = self.config['bidirectional']
        self.linear_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.data = self.config['data']
        self.starting_sample = self.config['starting_sample']
        self.starting_dist = self.config['starting_dist']
        self.act_embedding_dim = self.config['act_embedding_dim']
        self.device =1
        self.model_save_interval = self.config['model_save_interval']
        self.eval = eval
        self.test_data_path = self.config['test_data_path']
        self.test_data = np.loadtxt(self.test_data_path)
        self.pre_pos_count_embedding_dim=8
        self.stay_time_embedding_dim=8
        self.process_num = 45
        self.seed = seed
        os.makedirs(f'./results/result_{self.seed}_{self.noise}_{self.beta}/', exist_ok=True)
        os.makedirs(f'./results/result_{self.seed}_{self.noise}_{self.beta}/evals/', exist_ok=True)
          
        self.policy_net = ATNetwork(
            self.total_locations,
            self.time_scale,
            self.embedding_net,
            self.loc_embedding_dim,
            self.tim_embedding_dim,
            self.pre_pos_count_embedding_dim,
            self.stay_time_embedding_dim,
            self.hidden_dim,
            self.bidirectional,
            self.data,
            self.device,
            self.starting_sample,
            self.starting_dist,
            return_prob=True
        ).cuda()

        self.value_net = ATNetwork(
            self.total_locations,
            self.time_scale,
            self.embedding_net,
            self.loc_embedding_dim,
            self.tim_embedding_dim,
            self.pre_pos_count_embedding_dim,
            self.stay_time_embedding_dim,
            self.hidden_dim,
            self.bidirectional,
            self.data,
            self.device,
            self.starting_sample,
            self.starting_dist,
            return_prob=False
        ).cuda()
        self.discriminator = []
        for _ in range(len(self.file)):
            model = Discriminator(
            self.total_locations,
            self.time_scale,
            self.loc_embedding_dim,
            self.tim_embedding_dim,
            self.pre_pos_count_embedding_dim,
            self.stay_time_embedding_dim,
            self.act_embedding_dim,
            self.hidden_dim)
            self.discriminator.append(model)
        self.buffer = replay_buffer(self.capacity, self.gamma, self.lam)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001, weight_decay=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=0.001, weight_decay=1e-5)
        self.discriminator_optimizer = []
        for i in range(len(self.file)):
            self.discriminator_optimizer.append(torch.optim.Adam(self.discriminator[i].parameters(), lr=0.001))
        #loss
        self.disc_loss_func = nn.BCELoss()
        if self.eval:
            self.policy_net.load_state_dict(torch.load(f'./results/result_{self.seed}{self.noise}_{self.beta}/models_{1}/policy_net.pkl'))
            self.value_net.load_state_dict(torch.load(f'./results/result_{self.seed}_{self.noise}_{self.beta}/models_{1}/policy_net.pkl'))
            for i in range(len(self.file)):
                self.discriminator[i].load_state_dict(torch.load(f'./results/result_{self.seed}_{self.noise}_{self.beta}/models_{1}/discriminator_{i}.pth'))



    def sample_real_data(self, user_id):
        total_track_num = self.file[user_id].shape[0]*(self.file[user_id].shape[1]-1)
        sample_index = list(np.random.choice(total_track_num, self.batch_size))
        sample_index = [(x // (self.file[user_id].shape[1] - 1), x % (self.file[user_id].shape[1] - 1)) for x in sample_index]
        time = [index[1] for index in sample_index]
        pos = [self.file[user_id][index] for index in sample_index]
        next_pos = [self.file[user_id][index[0], index[1] + 1] for index in sample_index]
        history_pos = [list(self.file[user_id][index[0], :index[1] + 1]) for index in sample_index]
        pre_pos_count = [len(list(set(hp))) for hp in history_pos]
        stay_time = []
        for i in range(self.batch_size):
            st = 0
            for p in reversed(history_pos[i]):
                if p == next_pos[i]:
                    st += 1
                else:
                    break
            stay_time.append(st)
        home_pos = [self.file[user_id][index[0], 0] for index in sample_index]
        action = []
        for i in range(self.batch_size):
            if next_pos[i] == pos[i]:
                action.append(0)
            elif next_pos[i] == home_pos[i]:
                action.append(2)
            elif next_pos[i] in history_pos[i]:
                action.append(3)
            else:
                action.append(1)
        return list(zip(pos, time)), action, pre_pos_count, stay_time
    
    def ppo_train(self):
        pos, times, history_pos, home_point, pre_pos_count, stay_time, actions, returns, advantages = self.buffer.sample(self.batch_size)
        pos = torch.LongTensor(pos).cuda()
        times = torch.LongTensor(times).cuda()
        history_pos = torch.LongTensor(history_pos).cuda()
        home_point = torch.LongTensor(home_point).cuda()
        stay_time = torch.LongTensor(stay_time).cuda()
        pre_pos_count = torch.LongTensor(pre_pos_count).cuda() 
        advantages = torch.FloatTensor(advantages).unsqueeze(1).cuda()
        advantages = (advantages - advantages.mean()) / advantages.std()
        advantages = advantages.detach()
        returns = torch.FloatTensor(returns).cuda().unsqueeze(1).detach()

        for _ in range(self.value_iter):
            values = self.value_net.forward(pos, times, pre_pos_count, stay_time)
            value_loss = (returns - values).pow(2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        actions_d = torch.LongTensor(actions).unsqueeze(1).cuda()
        old_probs = self.policy_net.forward(pos, times, pre_pos_count, stay_time)
        old_probs = old_probs.gather(1, actions_d)
        dist = torch.distributions.Categorical(old_probs)
        entropy = dist.entropy().unsqueeze(1)

        for _ in range(self.policy_iter):
            probs = self.policy_net.forward(pos, times, pre_pos_count, stay_time)
            probs = probs.gather(1, actions_d)
            ratio = probs / old_probs.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * advantages
            policy_loss = - torch.min(surr1, surr2) - self.entropy_weight * entropy
            policy_loss = policy_loss.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad)
            self.policy_optimizer.step()
            
            
            
    def get_sample_data(self):
        dis_sample={}
        buffer_sample={}
        for i,user_id in enumerate(self.file):
            dis_sample[i] = self.sample_real_data(user_id)
            pos, times, history_pos, home_point, pre_pos_count, stay_time, actions, _, _ = self.buffer.sample(self.batch_size)
            buffer_sample[i] = (pos, times, history_pos, home_point, pre_pos_count, stay_time, actions)
        self.dis_sample = dis_sample
        self.buffer_sample = buffer_sample
        #return dis_sample, buffer_sample
    def discriminator_train(self):
        self.discriminator = [self.discriminator[i].cuda() for i in range(len(self.file))]
        for i,user_id in enumerate(self.file):
            expert_batch = self.sample_real_data(user_id)
            expert_observations, expert_actions, expert_pre_pos_count, expert_stay_time = expert_batch[0], expert_batch[1], expert_batch[2], expert_batch[3]
            expert_observations = np.vstack(expert_observations)
            expert_observations = torch.LongTensor(expert_observations).cuda()
            expert_actions_index = torch.LongTensor(expert_actions).unsqueeze(1).cuda()
            expert_stay_time = torch.LongTensor(expert_stay_time).unsqueeze(1).cuda()
            expert_pre_pos_count = torch.LongTensor(expert_pre_pos_count).unsqueeze(1).cuda()
            expert_trajs = torch.cat([expert_observations, expert_actions_index, expert_pre_pos_count, expert_stay_time], 1)
            expert_labels = torch.FloatTensor(self.batch_size, 1).fill_(0.0).cuda()
            pos, times, history_pos, home_point, pre_pos_count, stay_time, actions, _, _ = self.buffer.sample(self.batch_size)
            pos = torch.LongTensor(pos).view(-1, 1).cuda()
            times = torch.LongTensor(times).view(-1, 1).cuda()
            observations = torch.cat([pos, times], dim=-1).cuda()
            actions_index = torch.LongTensor(actions).unsqueeze(1).cuda()
            stay_time = torch.LongTensor(stay_time).unsqueeze(1).cuda()
            pre_pos_count = torch.LongTensor(pre_pos_count).unsqueeze(1).cuda()
            trajs = torch.cat([observations, actions_index, pre_pos_count, stay_time], 1)
            labels = torch.FloatTensor(self.batch_size, 1).fill_(1.0).cuda()
            for _ in range(self.disc_iter):
                # * optimize discriminator
                expert_reward = self.discriminator[i].forward(expert_trajs[:, 0], expert_trajs[:, 1], expert_trajs[:, 2], expert_trajs[:, 3], expert_trajs[:, 4])
                current_reward = self.discriminator[i].forward(trajs[:, 0], trajs[:, 1], trajs[:, 2], trajs[:, 3], trajs[:, 4])
                expert_loss = self.disc_loss_func(expert_reward, expert_labels)
                current_loss = self.disc_loss_func(current_reward, labels)
                loss = (expert_loss + current_loss) / 2
                self.discriminator_optimizer[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator[i].parameters(), self.clip_grad)
                self.discriminator_optimizer[i].step() 
        self.discriminator = [self.discriminator[i].cpu() for i in range(len(self.file))]
        
        
    def one_reward(self, data_list, reward_input_dict, result_queue):
        all_result={}
        for i in data_list:
            (pos, time, action, pre_pos_count, stay_time) = reward_input_dict[i]
            total_reward = 0
            for j in range(len(self.file)):
                total_reward += self.discriminator[j].forward(pos, time, action, pre_pos_count, stay_time)
            if self.noise == 0:
                reward = total_reward/len(self.file)
            else:
                reward = total_reward/len(self.file) + torch.from_numpy(np.random.laplace(0, self.noise, 1))
            mean_std =  []
            for _ in range(200):
                sample_result = random.sample(list(range(0,len(self.file))),self.alpha)
                outputs = [self.discriminator[u].forward(pos, time, action, pre_pos_count, stay_time).item() for u in sample_result]
                mean_std.append(outputs)
            value_c = (np.abs(np.random.laplace(0, self.noise*2, 1) + np.var(mean_std)))**0.5
            reward = torch.abs(reward - self.beta * torch.from_numpy(value_c))
            log_reward = - reward.log()
            all_result[i] = log_reward.detach().item()
        result_queue.put(all_result)

    def get_reward(self,reward_input_dict):
        length = len(reward_input_dict)
        iter_list = np.arange(length-length%self.process_num).reshape(self.process_num,-1).tolist()
        iter_list[-1].extend(np.arange(length-length%self.process_num,length).tolist())
        result_queue = Manager().Queue()
        reward_dict = {}
        processes = list()
        for i in range(self.process_num):
            p = Process(target=self.one_reward, args=(iter_list[i],reward_input_dict,result_queue))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for i in range(result_queue.qsize()):
            reward_dict.update(result_queue.get())
        return reward_dict

    def get_buffer(self,reward_dict,buffer_input_dict):
        for i in buffer_input_dict:
            (pos, time, history_pos, home_point, pre_pos_count, stay_time, action, done, value)=buffer_input_dict[i]
            custom_reward = reward_dict[i]
            self.buffer.store(pos, time, history_pos, home_point, pre_pos_count, stay_time, action, custom_reward, done, value)
        
        
    def eval_test(self, index):
        result = np.zeros_like(self.test_data)
        for i in tqdm(range(len(self.test_data))):
            t = 0
            pos = self.test_data[i][t]
            home_point = self.test_data[i][0]
            history_pos = []
            stay_time = 0
            pre_pos_count = 1
            self.env.set_state(pos=int(pos), t=int(t))
            result[i][t] = pos
            while True:
                action = self.policy_net.act(torch.LongTensor(np.expand_dims(pos, 0)).cuda(), torch.LongTensor(np.expand_dims(t, 0)).cuda(), torch.LongTensor(np.expand_dims(pre_pos_count, 0)).cuda(), torch.LongTensor(np.expand_dims(stay_time, 0)).cuda())
                next_pos, next_t, done, history_pos, home_point, pre_pos_count, stay_time = self.env.step(action, history_pos, home_point)
                result[i][next_t] = next_pos
                pos = next_pos
                t = next_t
                if done:
                    break
        np.savetxt(f'./results/result_{self.seed}_{self.noise}_{self.beta}/evals/eval_{index}.txt',result, fmt = '%d')

    def eval_data(self):
        self.eval_test(1)
        
    def run(self):
        setproctitle.setproctitle('gail@gaochangzheng')
        set_seed(self.seed)
        reward_input_dict=dict()
        buffer_input_dict=dict()
        buffer_count = 0 
        for i in range(self.episode):
            pos, time, history_pos, home_point, pre_pos_count, stay_time = self.env.reset()
            while True:
                action = self.policy_net.act(torch.LongTensor(np.expand_dims(pos, 0)).cuda(), torch.LongTensor(np.expand_dims(time, 0)).cuda(), torch.LongTensor(np.expand_dims(pre_pos_count, 0)).cuda(), torch.LongTensor(np.expand_dims(stay_time, 0)).cuda())
                next_pos, next_time, done, next_history_pos, next_home_point, next_pre_pos_count, next_stay_time= self.env.step(action, history_pos, home_point)
                value = self.value_net.forward(torch.LongTensor(np.expand_dims(pos, 0)).cuda(), torch.LongTensor(np.expand_dims(time, 0)).cuda(), torch.LongTensor(np.expand_dims(pre_pos_count, 0)).cuda(), torch.LongTensor(np.expand_dims(stay_time, 0)).cuda()).detach().item()
                reward_input_dict[buffer_count]=(torch.LongTensor([pos]).cpu(), 
                                                torch.LongTensor([time]).cpu(),
                                                torch.LongTensor([action]).cpu(),
                                                torch.LongTensor([pre_pos_count]).cpu(),
                                                torch.LongTensor([stay_time]).cpu())
                buffer_input_dict[buffer_count]=(pos, time, history_pos, home_point, pre_pos_count, stay_time, action,done, value)
                buffer_count += 1
                pos = next_pos
                time = next_time
                history_pos = next_history_pos
                home_point = next_home_point
                pre_pos_count = next_pre_pos_count
                stay_time = next_stay_time
                if done:
                    if buffer_count >= self.train_iter:
                        rewards1= self.get_reward(reward_input_dict)
                        self.get_buffer(rewards1,buffer_input_dict)
                        self.buffer.process()
                        self.discriminator_train()
                        self.ppo_train()
                        self.buffer.clear()
                        buffer_count = 0
                        reward_input_dict.clear()
                        buffer_input_dict.clear()
                        total_reward = sum([rewards1[i] for i in range(94,141)])
                        print('episode: {}'.format(i + 1), 'total_reward', total_reward)
                    break

            if (i + 1) % self.model_save_interval == 0:
                save_index = (i + 1) // self.model_save_interval
                os.makedirs(f'./results/result_{self.seed}_{self.noise}_{self.beta}/models_{save_index}', exist_ok=True)
                torch.save(self.policy_net.state_dict(), f'./results/result_{self.seed}_{self.noise}_{self.beta}/models_{save_index}/policy_net.pkl')
                torch.save(self.value_net.state_dict(), f'./results/result_{self.seed}_{self.noise}_{self.beta}/models_{save_index}/value_net.pkl')
                for idx, item in enumerate(self.discriminator):
                    torch.save(item.state_dict(), f'./results/result_{self.seed}_{self.noise}_{self.beta}/models_{save_index}/discriminator_{idx}.pth')
                self.eval_test(save_index)
