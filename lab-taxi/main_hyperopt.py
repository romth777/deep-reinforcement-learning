from agent import Agent
from monitor import interact
import gym
import numpy as np
from hyperopt import hp, tpe, Trials, fmin


hyperopt_parameters = {
    'alpha': hp.uniform('alpha', 0, 1)
    # 'gamma': hp.uniform('gamma', 0, 1)
    # 'eps': hp.uniform('eps', 0, 1)
}


def objective(args):
    env = gym.make('Taxi-v2')
    agent = Agent(alpha=args['alpha'])
    avg_rewards, best_avg_reward = interact(env, agent)
    return -1*best_avg_reward


# iterationする回数
max_evals = 200
# 試行の過程を記録するインスタンス
trials = Trials()

best = fmin(
    # 最小化する値を定義した関数
    objective,
    # 探索するパラメータのdictもしくはlist
    hyperopt_parameters,
    # どのロジックを利用するか、基本的にはtpe.suggestでok
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
    # 試行の過程を出力
    verbose=1
)

print(best)
print(trials.best_trial['result'])
