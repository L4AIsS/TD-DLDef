import numpy as np
import random


class ThompsonSampling:
    def __init__(self, layers, alpha, beta):
        """
            初始化多臂汤普森采样算法
        """
        self.__layers = layers # 存储层的名字
        self.__dict = {}
        for layer in layers:
            self.__dict[layer] = [alpha, beta] # 成功次数 & 失败次数

        self.record = {}             # 记录结果和次数，包括 <A,B> 和 A 类型的结果，用来判断 reward
        self.__single_record = {}    # 记录 A 类型的结果

        self.__pre = "" # 上一次的采样结果
        self.__cnt = 0

        # 层的lines和branch
        self.__layers_branch_lines = np.load('utils/layer_branch_lines.npy', allow_pickle=True).item()
        # self.__layers_branch_lines = {}

    def choose_arm(self, candidates):
        """
        选择臂：从每个臂的后验分布中采样，选择值最大的臂
        """
        c_cnt = len(candidates)
        successes = np.ones(c_cnt)
        failures = np.ones(c_cnt)
        for i, candidate in enumerate(candidates):
            successes[i] += self.__dict[candidate][0]
            failures[i] += self.__dict[candidate][1]

        sampled_values = np.random.beta(successes, failures) # 从每个臂的Beta分布中采样

        # 选择采样值最大的臂
        re = np.argmax(sampled_values)
        name = candidates[int(re)]

        # # 设置随机采样
        re = random.randint(0, c_cnt - 1)
        name = candidates[re]

        return name

    def update(self, chosen_name, reward):
        """
        更新所选臂的成功和失败次数
        :param chosen_name: 选择的臂
        :param reward: 奖励（0 或 1）
        """
        if self.__single_record.get(chosen_name, None) is None:
            self.__single_record[chosen_name] = 1
        else:
            self.__single_record[chosen_name] += 1

        self.__pre = chosen_name
        self.__cnt += 1

        if reward == 1:
            self.__dict[chosen_name][1] += 6
        else:
            self.__dict[chosen_name][0] += 6


    def is_reward(self, selected_name):

        selected_cnt = self.__single_record.get(selected_name, None)
        if selected_cnt is None:
            return 1

        if selected_cnt / self.__cnt < 1 / len(self.__layers):
            return 1
        return 0

    def print(self):
        for layer in self.__layers:
            print(f'{layer} 次数 {self.__single_record.get(layer, None)}')
        print(self.__single_record)

    def get_single_record(self):
        return self.__single_record

    def cal_coverage(self):
        return round(100 * len(self.__single_record) / len(self.__layers), 2), self.__single_record

    def cal_branch_and_lines(self):
        branch_cnt = 0
        lines_cnt = 0
        for layer in self.__single_record:
            branch_line = self.__layers_branch_lines.get(layer, None)
            if branch_line is not None:
                branch_cnt += branch_line[0]
                lines_cnt += branch_line[1]
        return branch_cnt, lines_cnt


if __name__ == '__main__':
    layers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    a_sampling = ThompsonSampling(layers=layers, alpha=1, beta=1)
    for i in range(100):
        cands = random.sample(layers, 5)
        re = a_sampling.choose_arm(cands)
        print(f'选择的结果 {re}')
        a_sampling.update(re, a_sampling.is_reward(re))
    a_sampling.print()
    print(a_sampling.cal_coverage())

# random 84%
