import gym
import slimevolleygym
 
env = gym.make('SlimeVolley-v0')
observation = env.reset()
done = False
while not done:
    action = env.action_space.sample() # 随机选择动作
    observation, reward, done, info = env.step(action)
    env.render() # 渲染环境，观察结果
env.close()
