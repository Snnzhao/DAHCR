
import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
# sys.path.append('..')

from collections import namedtuple
import argparse
from itertools import count, chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from sum_tree import SumTree

#TODO select env
from RL.env_binary_question import BinaryRecommendEnv
from RL.env_enumerated_question import EnumeratedRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from gcn import GraphEncoder
import time
import warnings
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM: BinaryRecommendEnv,
    LAST_FM_STAR: BinaryRecommendEnv,
    YELP: EnumeratedRecommendEnv,
    YELP_STAR: BinaryRecommendEnv
    }
FeatureDict = {
    LAST_FM: 'feature',
    LAST_FM_STAR: 'feature',
    YELP: 'large_feature',
    YELP_STAR: 'feature'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_cand_feature', 'next_cand_item','scpr','next_scpr','ac_label','nac_label'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemoryPER(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity, a = 0.6, e = 0.01):
        self.tree =  SumTree(capacity)
        self.capacity = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        
    def push(self, *args):
        data = Transition(*args)
        p = (np.abs(self.prio_max) + self.e) ** self.a #  proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        batch_data = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            batch_data.append(data)
            priorities.append(p)
            idxs.append(idx)
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return idxs, batch_data, is_weight
    
    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p) 
        
    def __len__(self):
        return self.tree.n_entries


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=100+25,jiexian=None):
        super(DQN, self).__init__()
        self.device='cuda'
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(hidden_size, 2)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        # V(s)
        self.fc2_value = nn.Linear(hidden_size, hidden_size)
        self.out_value = nn.Linear(hidden_size, 1)
        # Q(s,a)
        self.fc2_advantage = nn.Linear(hidden_size + action_size, hidden_size)   
        self.out_advantage = nn.Linear(hidden_size, 1)
        self.jiexian=jiexian

    def forward(self, x, y, len_feature=None, action_batch=None, choose_action=True, turn=0):
        """
        :param x: encode history [N*L*D]; y: action embedding [N*K*D]
        :return: v: action score [N*K]
        """
        # print(x.shape,y.shape)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.out(x).squeeze(1)
        # action_value_ = torch.softmax(out,dim=-1).squeeze(1)
        # cat = RelaxedOneHotCategorical(0.3, logits=action_value_)
        logit_action_value_ = torch.softmax(out,dim=-1).squeeze(1)
        cat = RelaxedOneHotCategorical(0.3, logits=logit_action_value_)
        action_value_ = cat.rsample().float()
        # if action_value_[:,0]>action_value_[:,1]:
        #     action_value=torch.tensor([[1,0]],device=self.device)
        # else:
        #     action_value=torch.tensor([[0,1]],device=self.device)
        action_value=torch.zeros_like(action_value_,device=self.device)
        action_value[:,0]=torch.where(action_value_[:,0]>action_value_[:,1],1,0)
        action_value[:,1]=torch.where(action_value[:,0]==1,0,1)
        # V(s)
        #print(action_value)
        value = self.out_value(F.relu(self.fc2_value(x))).squeeze(dim=2) #[N*1*1]
        # Q(s,a)
        if choose_action:
            x = x.repeat(1, y.size(1), 1)
        state_cat_action = torch.cat((x,y),dim=2)
        advantage = self.out_advantage(F.relu(self.fc2_advantage(state_cat_action))).squeeze(dim=2) #[N*K]

        if choose_action:
            qsa = advantage + value - advantage.mean(dim=1, keepdim=True)
            # print(qsa.shape,action_value.shape)


            qsa[:,:len_feature]*= action_value_[:,0].unsqueeze(-1)#+=(1-action_value[:,0].unsqueeze(-1))*(-1e10)
                # item
            qsa[:,len_feature:]*= action_value_[:,1].unsqueeze(-1)#+=(1-action_value[:,1].unsqueeze(-1))*(-1e10)
        else:
            qsa = advantage + value
            # print(action_batch.shape,action_value.shape)
            v=action_value.gather(1,torch.where(action_batch<self.jiexian,1,0))
            # print(v.shape)
            qsa=qsa*v
        # print(qsa.shape)
        return action_value_, qsa,out


class Agent(object):
    def __init__(self, device, memory, state_size, action_size, hidden_size, gcn_net, learning_rate, l2_norm, PADDING_ID,jiexian=None, EPS_START = 0.9, EPS_END = 0.1, EPS_DECAY = 0.0001, tau=0.01):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.device = device
        self.gcn_net = gcn_net
        self.policy_net = DQN(state_size, action_size, hidden_size+25,jiexian).to(device)
        self.target_net = DQN(state_size, action_size, hidden_size+25,jiexian).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(chain(self.policy_net.parameters(),self.gcn_net.parameters()), lr=learning_rate, weight_decay = l2_norm)
        self.memory = memory
        self.loss_func = nn.MSELoss()
        self.PADDING_ID = PADDING_ID
        self.tau = tau
        self.jiexian=jiexian
        self.criterion=nn.CrossEntropyLoss()


    def select_action(self, scpr,state, cand_feature, cand_item, action_space, is_test=False, is_last_turn=False, turn=0):
        state_emb = self.gcn_net([state])
        scpr_emb=torch.tensor(scpr).view(state_emb.shape[0], state_emb.shape[1],-1).cuda()
        state_emb=torch.cat((scpr_emb, state_emb),-1)
        cand=cand_feature+cand_item
        len_feature=len(cand_feature)
        cand = torch.LongTensor([cand]).to(self.device)
        cand_emb = self.gcn_net.embedding(cand)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY) # forever 0.1
        self.steps_done += 1
        if is_test or sample > eps_threshold: #predict with experience
            if is_test and (len(action_space[1]) <= 10 or is_last_turn):
                return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space[1], False
            with torch.no_grad():
                _,actions_value,out = self.policy_net(state_emb, cand_emb, len_feature, turn=turn)
                print(sorted(list(zip(cand[0].tolist(), actions_value[0].tolist())), key=lambda x: x[1], reverse=True))
                action = cand[0][actions_value.argmax().item()]

                sorted_actions = cand[0][actions_value.sort(1, True)[1].tolist()]
                return action, sorted_actions.tolist(), False
        else: #explore
            shuffled_cand = action_space[0]+action_space[1]#item + feature
            random.shuffle(shuffled_cand)
            return torch.tensor(shuffled_cand[0], device=self.device, dtype=torch.long), shuffled_cand, True
    
    def update_target_model(self):
        #soft assign
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    def optimize_model(self, BATCH_SIZE, GAMMA,epoch):
        if len(self.memory) < BATCH_SIZE:
            return
        self.update_target_model()

        idxs, transitions, is_weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_emb_batch = self.gcn_net(list(batch.state))
        scpr_emb_batch=torch.stack(list(batch.scpr)).view(state_emb_batch.shape[0], state_emb_batch.shape[1], -1)
        state_emb_batch=torch.cat((scpr_emb_batch, state_emb_batch), -1)
        try:
            action_batch = torch.LongTensor(np.array(batch.action).astype(int).reshape(-1, 1)).to(self.device) #[N*1]
        except:
            print(batch.action)
            raise ValueError
        action_emb_batch = self.gcn_net.embedding(action_batch)
        reward_batch = torch.FloatTensor(np.array(batch.reward).astype(float).reshape(-1, 1)).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = []
        n_cands_feature = []
        n_cands_item = []
        nac_label=[]
        n_scpr_state=[]
        for s, cf,ci, l, scp in zip(batch.next_state, batch.next_cand_feature,batch.next_cand_item,batch.nac_label, batch.next_scpr):
            if s is not None:
                n_states.append(s)
                n_cands_feature.append(cf)
                n_cands_item.append(ci)
                nac_label.append(l)
                n_scpr_state.append(scp)
        nac_label=torch.tensor(nac_label,device=self.device)

        next_state_emb_batch = self.gcn_net(n_states)


        next_scpr_emb_batch = torch.stack(n_scpr_state).view(next_state_emb_batch.shape[0], next_state_emb_batch.shape[1], -1)
        next_state_emb_batch = torch.cat((next_scpr_emb_batch, next_state_emb_batch), -1)

        next_cand_feature_batch = self.padding(n_cands_feature)
        next_cand_item_batch = self.padding(n_cands_item)
        next_cand_batch=torch.cat((next_cand_feature_batch, next_cand_item_batch), dim=-1)
        next_cand_emb_batch = self.gcn_net.embedding(next_cand_batch)
        next_len_feature=next_cand_feature_batch.shape[-1]
        action_value,q_eval,out = self.policy_net(state_emb_batch, action_emb_batch, action_batch=action_batch, choose_action=False)
        # XXX
        state_action_value=action_value.gather(1, torch.where(action_batch < self.jiexian, 1, 0))
        # state_action_value=out.gather(1, torch.where(action_batch < self.jiexian, 1, 0))
        # Double DQN
        best_actions = torch.gather(input=next_cand_batch, dim=1, index=self.policy_net(next_state_emb_batch, next_cand_emb_batch, next_len_feature)[1].argmax(dim=1).view(len(n_states),1).to(self.device))
        best_actions_emb = self.gcn_net.embedding(best_actions)
        q_target = torch.zeros((BATCH_SIZE,1), device=self.device)
        # print(non_final_mask.shape)
        action_value_n,qqq,out_n = self.target_net(next_state_emb_batch,best_actions_emb,action_batch=best_actions,choose_action=False)
        q_target[non_final_mask]=qqq.detach()
        q_target = reward_batch + GAMMA * q_target
        next_state_values=torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] =action_value_n.max(1)[0].detach()
        # XXX
        # next_state_values[non_final_mask] =out_n.max(1)[0].detach()
        state_value_target=reward_batch + GAMMA * next_state_values
        # prioritized experience replay
        errors = (q_eval - q_target).detach().cpu().squeeze().tolist()
        self.memory.update(idxs, errors)

        ac_label=torch.tensor(batch.ac_label).to(self.device)
        pre_score = torch.cat([action_value, action_value_n], dim=0)
        ground = torch.cat([ac_label, nac_label], dim=-1)
        # pre=pre_score.gather(1,ground.unsqueeze(0)).view(-1)
        # print(pre_score.shape,ground.shape)
        a_loss = self.criterion(pre_score, ground)
        # mean squared error loss to minimize
        if epoch==0:
            loss=a_loss + (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(state_action_value, state_value_target)).mean()
            # loss=a_loss
        else:
            print('loss:',(torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean())
            # loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean() + (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(state_action_value, state_value_target)).mean()
            loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.data
    def save_model(self, data_name, filename, epoch_user):
        save_rl_agent(dataset=data_name, model={'policy': self.policy_net.state_dict(), 'gcn': self.gcn_net.state_dict()}, filename=filename, epoch_user=epoch_user)
    def load_model(self, data_name, filename, epoch_user):
        model_dict = load_rl_agent(dataset=data_name, filename=filename, epoch_user=epoch_user)
        self.policy_net.load_state_dict(model_dict['policy'])
        self.gcn_net.load_state_dict(model_dict['gcn'])
    
    def padding(self, cand):
        pad_size = max([len(c) for c in cand])
        padded_cand = []
        for c in cand:
            cur_size = len(c)
            new_c = np.ones((pad_size)) * self.PADDING_ID
            new_c[:cur_size] = c
            padded_cand.append(new_c)
        return torch.LongTensor(padded_cand).to(self.device)


def train(args, kg, dataset, filename):
    env = EnvDict[args.data_name](kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn, cand_num=args.cand_num, cand_item_num=args.cand_item_num,
                       attr_num=args.attr_num, mode='train', ask_num=args.ask_num, entropy_way=args.entropy_method, fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    memory = ReplayMemoryPER(args.memory_size) #50000
    embed = torch.FloatTensor(np.concatenate((env.ui_embeds, env.feature_emb, np.zeros((1,env.ui_embeds.shape[1]))), axis=0))
    gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg, embeddings=embed, \
        fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.hidden).to(args.device)
    jiexian=len(env.ui_embeds)
    agent = Agent(device=args.device, memory=memory, state_size=args.hidden, action_size=embed.size(1), \
        hidden_size=args.hidden, gcn_net=gcn_net, learning_rate=args.learning_rate, l2_norm=args.l2_norm, PADDING_ID=embed.size(0)-1,jiexian=jiexian)

    #ealuation metric  ST@T
    #agent load policy parameters
    if args.load_rl_epoch != 0 :
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)

    test_performance = []
    if args.eval_num == 1:
        SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, 0)
        test_performance.append(SR15_mean)
    for train_step in range(1, args.max_steps+1):
        SR5, SR10, SR15, AvgT, Rank, total_reward = 0., 0., 0., 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            print('\n================new tuple:{}===================='.format(i_episode))
            if not args.fix_emb:
                scpr,state, cand_feature, cand_item, ac_label, action_space = env.reset(agent.gcn_net.embedding.weight.data.cpu().detach().numpy())  # Reset environment and record the starting state
            else:
                scpr,state, cand_feature, cand_item, ac_label, action_space = env.reset()
            #state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
            epi_reward = 0
            is_last_turn = False
            for t in count():   # user  dialog # 0-14
                if t == 14:
                    is_last_turn = True
                action, sorted_actions,is_sample = agent.select_action(scpr,state, cand_feature, cand_item, action_space, is_last_turn=is_last_turn, turn=t)
                if not args.fix_emb:
                    next_scpr,next_state, next_cand_feature, next_cand_item,nac_label, action_space, reward, done = env.step(action.item(), sorted_actions, agent.gcn_net.embedding.weight.data.cpu().detach().numpy(), is_sample=is_sample)
                else:
                    next_scpr,next_state, next_cand_feature, next_cand_item,nac_label, action_space, reward, done = env.step(action.item(), sorted_actions, is_sample=is_sample)
                epi_reward += reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
                if done:
                    next_state = None

                agent.memory.push(state, action.cpu(), next_state, reward.cpu(), next_cand_feature, next_cand_item,torch.tensor(scpr).cuda(),torch.tensor(next_scpr).cuda(),torch.tensor([int(ac_label)],device=args.device),torch.tensor([int(nac_label)],device=args.device))
                state = next_state
                cand_feature = next_cand_feature
                cand_item = next_cand_item
                scpr=next_scpr
                ac_label=nac_label
                newloss1 = agent.optimize_model(args.batch_size, args.gamma,0)
                newloss2 = agent.optimize_model(args.batch_size, args.gamma,1)
                if newloss1 is not None:
                    loss += newloss1
                if newloss2 is not None:
                    loss += newloss2
                if done:
                    # every episode update the target model to be same with model
                    if reward.item() == 1:  # recommend successfully
                        if t < 5:
                            SR5 += 1
                            SR10 += 1
                            SR15 += 1
                        elif t < 10:
                            SR10 += 1
                            SR15 += 1
                        else:
                            SR15 += 1
                        Rank += (1/math.log(t+3,2) + (1/math.log(t+2,2)-1/math.log(t+3,2))/math.log(done+1,2))
                    else:
                        Rank += 0
                    AvgT += t+1
                    total_reward += epi_reward
                    break
        enablePrint() # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, rewards:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.sample_times, SR10 / args.sample_times, SR15 / args.sample_times,
                                                AvgT / args.sample_times, Rank / args.sample_times, total_reward / args.sample_times, args.sample_times))

        if train_step % args.eval_num == 0:
            SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, train_step)
            test_performance.append(SR15_mean)
        if train_step % args.save_num == 0:
            agent.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
    print(test_performance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='l2 regularization.')
    parser.add_argument('--hidden', type=int, default=100, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default=LAST_FM_STAR, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    parser.add_argument('--entropy_method', type=str, default='weight_entropy', help='entropy_method is one of {entropy, weight entropy}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading RL model')

    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--max_steps', type=int, default=100, help='max training steps')
    parser.add_argument('--eval_num', type=int, default=10, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=10, help='the number of steps to save RL model and metric')
    parser.add_argument('--observe_num', type=int, default=500, help='the number of steps to print metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=10, help='candidate item sampling number')
    parser.add_argument('--fix_emb', action='store_false', help='fix embedding or not')
    parser.add_argument('--embed', type=str, default='transe', help='pretrained embeddings')
    parser.add_argument('--seq', type=str, default='transformer', choices=['rnn', 'transformer', 'mean'], help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    #reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)

    dataset = load_dataset(args.data_name)
    filename = 'train-data-{}-RL-cand_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    train(args, kg, dataset, filename)

if __name__ == '__main__':
    main()