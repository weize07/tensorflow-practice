from my_thread import *
from engine import *
from agent import *

totalTick = 0
def main():
    engine = FlappyEngine()
    agent = DQNAgent(engine)
    mthread = MyThread(engine)
    mthread.start()

    time.sleep(1)
    agent.ready()

    round = 0
    while True:
        round += 1
        print("round-", round, ": training...")
        train(engine, agent)
        print("round-", round, ": testing...")
        test(engine, agent)

def train(engine, agent):
    tick = 0
    while tick < 100:
        time.sleep(0.1)
        stateT = engine.state()
        action = agent.egreedy()
        stateT1 = engine.state()
        if stateT1[0] == S_DEAD:
            reward = -1
            agent.ready()
        else:
            reward = 1
        agent.memorize(stateT[1:], action, reward, stateT1[1:])
        if totalTick > 20:
            engine.pause()
            agent.train_Q_network()
            engine.resume()
        totalTick += 1
        tick += 1

def test(engine, agent):
    while True:
        time.sleep(0.1)
        agent.action()
        stateT1 = engine.state()
        if stateT1[0] == S_DEAD:
            reward = -1
            agent.ready()
        else:
            reward = 1
        agent.memorize(stateT[1:], action, reward, stateT1[1:])
        if totalTick > 20:
            engine.pause()
            agent.train_Q_network()
            engine.resume()
        totalTick += 1
        tick += 1



if __name__ == '__main__':
    main()
