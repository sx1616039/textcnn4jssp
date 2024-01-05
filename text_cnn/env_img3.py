import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import random

import pandas as pd


def get_optimal(job_dict, opt_sign):
    if opt_sign == "max":
        return max(job_dict.values())
    elif opt_sign == "min":
        return min(job_dict.values())
    elif opt_sign == "random":
        if len(job_dict) <= 1:
            return min(job_dict.values())
        ran = np.random.randint(0, len(job_dict))
        i = 0
        for k, v in job_dict.items():
            if i == ran:
                return v
            i += 1


class JobEnv:
    def __init__(self, case_name, path):
        self.PDRs = {"SPT": "min", "MWKR": "max", "FDD/MWKR": "min", "MOPNR": "max", "LRM": "max", "FIFO": "max",
                     "LPT": "max", "LWKR": "min", "FDD/LWKR": "max", "LOPNR": "min", "SRM": "min", "LIFO": "min"}
        self.pdr_label = ["SPT", "MWKR", "FDD/MWKR", "MOPNR", "LRM", "FIFO",
                          "LPT", "LWKR", "FDD/LWKR", "LOPNR", "SRM", "LIFO"]

        self.case_name = case_name
        file = path + case_name + ".txt"
        with open(file, 'r') as f:
            user_line = f.readline()
            user_line = str(user_line).replace('\n', '\t')
            data = user_line.split('\t')
            while data.__contains__(""):
                data.remove("")
            self.m_n = list(map(int, data))
            data = f.read()
            data = str(data).replace('\n', '\t')
            data = str(data).split('\t')
            while data.__contains__(""):
                data.remove("")
            job = list(map(int, data))

        self.job = np.array(job).reshape(self.m_n[0], self.m_n[1] * 2)
        self.job_num = self.m_n[0]
        self.machine_num = self.m_n[1]

        self.action_num = int(len(self.pdr_label)/2)

        self.current_time = 0  # current time
        self.next_time_on_machine = None
        self.job_on_machine = None
        self.current_op_of_job = None
        self.assignable_job = None
        self.finished_jobs = None

        self.result_dict = {}
        self.solution_table = None
        self.machine_table = None
        self.time_table = None

        self.max_op_len = 0
        # find maximum operation length of all jobs
        for j in range(self.job_num):
            for i in range(self.machine_num):
                if self.max_op_len < self.job[j][i * 2 + 1]:
                    self.max_op_len = self.job[j][i * 2 + 1]

        self.last_release_time = None
        self.done = False
        self.reward = 0
        self.scale = self.job_num*self.machine_num
        self.solution_cnt = 0

    def reset(self):
        self.current_time = 0  # current time
        self.next_time_on_machine = np.repeat(0, self.machine_num)
        self.job_on_machine = np.repeat(-1, self.machine_num)  # -1 implies idle machine
        self.assignable_job = np.ones(self.job_num, dtype=bool)  # whether a job is assignable
        self.last_release_time = np.repeat(0, self.job_num)
        self.current_op_of_job = np.repeat(0, self.job_num)  # current operation state of job
        self.finished_jobs = np.zeros(self.job_num, dtype=bool)

        self.machine_table = np.zeros([self.job_num, self.machine_num], dtype=float)
        self.time_table = np.zeros([self.job_num, self.machine_num], dtype=float)
        self.solution_table = np.zeros([self.job_num, self.machine_num], dtype=float)

        self.result_dict = {}
        self.done = False
        self.solution_cnt = 0
        return self._get_state()

    def get_feature(self, job_id, feature):
        if feature == self.pdr_label[0] or feature == self.pdr_label[6]:
            return self.job[job_id][self.current_op_of_job[job_id] * 2 + 1]
        elif feature == self.pdr_label[1] or feature == self.pdr_label[7]:
            work_remain = 0
            for i in range(self.machine_num-self.current_op_of_job[job_id]):
                work_remain += self.job[job_id][(i + self.current_op_of_job[job_id])*2+1]
            return work_remain
        elif feature == self.pdr_label[2] or feature == self.pdr_label[8]:
            work_remain = 0
            work_done = 0
            for i in range(self.machine_num - self.current_op_of_job[job_id]):
                work_remain += self.job[job_id][(i + self.current_op_of_job[job_id]) * 2 + 1]
            for k in range(self.current_op_of_job[job_id]):
                work_done += self.job[job_id][k * 2 + 1]
            if work_remain == 0:
                return 10000
            return work_done/work_remain
        elif feature == self.pdr_label[3] or feature == self.pdr_label[9]:
            return self.machine_num - self.current_op_of_job[job_id] + 1
        elif feature == self.pdr_label[4] or feature == self.pdr_label[10]:
            work_remain = 0
            for i in range(self.machine_num - self.current_op_of_job[job_id] - 1):
                work_remain += self.job[job_id][(i + self.current_op_of_job[job_id] + 1) * 2 + 1]
            return work_remain
        elif feature == self.pdr_label[5] or feature == self.pdr_label[11]:
            return self.current_time - self.last_release_time[job_id]
        return 0

    def _get_state(self):
        return np.array([self.machine_table, self.time_table, self.solution_table])

    def step(self, action):
        self.done = False
        self.reward = 0

        # action is PDR
        PDR = [self.pdr_label[action], self.PDRs.get(self.pdr_label[action])]
        # allocate jobs according to PDRs
        job_dict = {}
        for i in range(self.job_num):
            if self.assignable_job[i]:
                job_dict[i] = self.get_feature(i, PDR[0])
        if len(job_dict) > 0:
            for key in job_dict.keys():
                machine_id = self.job[key][self.current_op_of_job[key] * 2]
                if job_dict.get(key) == get_optimal(job_dict, PDR[1]) and self.job_on_machine[machine_id] < 0:
                    self.allocate_job(key)
                    break  # one step at one time
        state = self._get_state()
        if self.stop():
            self.done = True
        return state, self.reward/self.max_op_len, self.done

    def allocate_job(self, job_id):
        self.solution_cnt += 1
        stage = self.current_op_of_job[job_id]
        machine_id = self.job[job_id][stage * 2]
        process_time = self.job[job_id][stage * 2 + 1]

        self.time_table[job_id][self.current_op_of_job[job_id]] = process_time / self.max_op_len
        self.machine_table[job_id][self.current_op_of_job[job_id]] = machine_id / self.machine_num
        self.solution_table[job_id][self.current_op_of_job[job_id]] = self.solution_cnt / self.scale

        self.job_on_machine[machine_id] = job_id
        start_time = self.next_time_on_machine[machine_id]
        self.next_time_on_machine[machine_id] += process_time
        end_time = start_time + process_time
        self.result_dict[job_id + 1, machine_id + 1] = start_time, end_time, process_time

        self.last_release_time[job_id] = self.current_time
        self.assignable_job[job_id] = False
        # assignable jobs whose current machine are employed will not be assignable
        for x in range(self.job_num):
            if self.assignable_job[x] and self.job[x][self.current_op_of_job[x] * 2] == machine_id:
                self.assignable_job[x] = False
        # there is no assignable jobs after assigned a job and time advance is needed
        # self.reward -= process_time
        while sum(self.assignable_job) == 0 and not self.stop():
            self.reward -= self.time_advance()
            self.release_machine()

    def time_advance(self):
        hole_len = 0
        min_next_time = min(self.next_time_on_machine)
        if self.current_time < min_next_time:
            self.current_time = min_next_time
        else:
            self.current_time = self.find_second_min()
        for machine in range(self.machine_num):
            dist_need_to_advance = self.current_time - self.next_time_on_machine[machine]
            if dist_need_to_advance > 0:
                self.next_time_on_machine[machine] += dist_need_to_advance
                hole_len += dist_need_to_advance
            else:
                job = self.job_on_machine[machine]
                if self.current_op_of_job[job]<self.machine_num:
                    self.time_table[job][self.current_op_of_job[job]] = -dist_need_to_advance
        return hole_len

    def release_machine(self):
        for k in range(self.machine_num):
            cur_job_id = self.job_on_machine[k]
            if cur_job_id >= 0 and self.current_time >= self.next_time_on_machine[k]:
                self.job_on_machine[k] = -1
                self.last_release_time[cur_job_id] = self.current_time
                for x in range(self.job_num):  # release jobs on this machine
                    if not self.finished_jobs[x] and self.job[x][self.current_op_of_job[x] * 2] == k:
                        self.assignable_job[x] = True
                # self.state_table[cur_job_id][self.current_op_of_job[cur_job_id] * 2] = -self.machine_num
                self.current_op_of_job[cur_job_id] += 1
                if self.current_op_of_job[cur_job_id] >= self.machine_num:
                    self.finished_jobs[cur_job_id] = True
                    self.assignable_job[cur_job_id] = False
                else:
                    next_machine = self.job[cur_job_id][self.current_op_of_job[cur_job_id] * 2]
                    if self.job_on_machine[next_machine] >= 0:  # 如果下一工序的机器被占用，则作业不可分配
                        self.assignable_job[cur_job_id] = False

    def stop(self):
        if sum(self.current_op_of_job) < self.machine_num * self.job_num:
            return False
        return True

    def find_second_min(self):
        min_time = min(self.next_time_on_machine)
        second_min_value = 100000
        for value in self.next_time_on_machine:
            if min_time < value < second_min_value:
                second_min_value = value
        if second_min_value == 100000:
            return min_time
        return second_min_value

    def draw_gantt(self, file_name):
        font_dict = {
            "style": "oblique",
            "weight": "bold",
            "color": "white",
            "size": 14
        }
        machine_labels = [" "]  # 生成y轴标签
        for i in range(self.machine_num):
            machine_labels.append("machine " + str(i + 1))
        plt.figure(1)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
        colors = ['#%06X' % random.randint(0, 256 ** 3 - 1) for _ in range(800)]
        for k, v in self.result_dict.items():
            plt.barh(y=k[1] - 1, width=v[2], left=v[0], edgecolor="black", color=colors[round(k[0])])
            plt.text(((v[0] + v[1]) / 2), k[1] - 1, str(round(k[0])), fontdict=font_dict)
        plt.yticks([i - 1 for i in range(self.machine_num + 1)], machine_labels)
        plt.title(self.case_name)
        plt.xlabel("time")
        plt.ylabel("machine")
        if not os.path.exists('gantt'):
            os.makedirs('gantt')
        plt.savefig("gantt/" + file_name + ".png")
        plt.close()


if __name__ == '__main__':
    data_path = "../data_set_uncertain_time/"
    PDR_label = ["SPT", "MWKR", "FDD/MWKR", "MOPNR", "LRM", "FIFO", "LPT", "LWKR", "FDD/LWKR", "LOPNR", "SRM", "LIFO"]
    results = pd.DataFrame(columns=PDR_label, dtype=int)
    for file in os.listdir(data_path):
        title = file.split('.')[0]  # file name
        env = JobEnv(title, data_path)
        case_result = []
        for pdr in range(len(PDR_label)):
            env.reset()
            cnt = 0
            while not env.stop():
                cnt += 1
                env.step(pdr)
            case_result.append(str(env.current_time))
            # env.draw_gantt("PDR")
        results.loc[title] = case_result
        print(title + str(case_result))
    results.to_csv("PDR-data_set_uncertain_time.csv")
