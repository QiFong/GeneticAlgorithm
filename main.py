import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import datetime

DNA_SIZE = 24  # 编码长度
POP_SIZE = 100  # 种群大小
CROSS_RATE = 0.8  # 交叉率
MUTA_RATE = 0.2  # 变异率
Iterations = 100  # 代次数
X_BOUND = [0, 9]  # X区间


def F(x):  # 适应度函数
    return x + 10*sin(5*x) + 7*cos(4*x)


def decodeDNA(pop):  # 解码
    x = pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    return x


def getfitness(pop):
    x= decodeDNA(pop)
    temp = F(x)
    return (temp - np.min(temp)) + 0.0001  # 减去最小的适应度是为了防止适应度出现负数


def select(pop, fitness):  # 根据适应度选择
    temp = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=(fitness) / (fitness.sum()))
    return pop[temp]


def crossmuta(pop, CROSS_RATE):
    new_pop = []
    for i in pop:  # 遍历种群中的每一个个体，将该个体作为父代
        temp = i  # 子代先得到父亲的全部基因
        if np.random.rand() < CROSS_RATE:  # 以交叉概率发生交叉
            j = pop[np.random.randint(POP_SIZE)]  # 从种群中随机选择另一个个体，并将该个体作为母代
            cpoints1 = np.random.randint(0, DNA_SIZE - 1)  # 随机产生交叉的点
            cpoints2 = np.random.randint(cpoints1, DNA_SIZE)
            temp[cpoints1:cpoints2] = j[cpoints1:cpoints2]  # 子代得到位于交叉点后的母代的基因
        mutation(temp, MUTA_RATE)  # 后代以变异率发生变异
        new_pop.append(temp)
    return new_pop


def mutation(temp, MUTA_RATE):
    if np.random.rand() < MUTA_RATE:  # 以MUTA_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        temp[mutate_point] = temp[mutate_point] ^ 1  # 将变异点的二进制为反转


def print_info(pop):  # 用于输出结果
    fitness = getfitness(pop)
    maxfitness = np.argmax(fitness)  # 返回最大值的索引值
    print("max_fitness:", fitness[maxfitness])
    x= decodeDNA(pop)
    print("最优的基因型：", pop[maxfitness])
    print("(x):", (x[maxfitness]))
    print("F(x)_max = ", F(x[maxfitness]))


def plot(ax):
    X = np.linspace(*X_BOUND, 1000)
    Y = F(X)
    ax.plot(X,Y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


if __name__ == "__main__":
    start_t = datetime.datetime.now()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plot(ax)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))
    for _ in range(Iterations):  # 迭代N代
        x = decodeDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, F(x), c='black', marker='o')
        plt.show()
        plt.pause(0.1)
        pop = np.array(crossmuta(pop, CROSS_RATE))
        fitness = getfitness(pop)
        pop = select(pop, fitness)  # 选择生成新的种群
    end_t = datetime.datetime.now()
    print((end_t - start_t).seconds)
    print_info(pop)
    plt.ioff()
    plot(ax)