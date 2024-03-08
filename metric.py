import pprint

class Metrics:
    def __init__(self):
        self.success = 0
        self.failed = 0
        self.total_reward = 0
        self.iteration = 0
        self.metrics = []

    def recordIteration(self, reward, isSuccess, step):
        if isSuccess:
            self.success += 1
        else:
            self.failed += 1
        self.iteration += 1
        self.total_reward += reward

        self.metrics.append([isSuccess, reward, step])
    
    def printMetrics(self):
        for i in range(len(self.metrics)):
            wf = "Win" if self.metrics[i][0] else "Fail"
            print(f"--- Iteration {i} --- ")
            pprint.pprint(f"Terminal State: {wf}")
            pprint.pprint(f"Reward: {self.metrics[i][1]}")
            pprint.pprint(f"Step: {self.metrics[i][2]}")
            print()
        
        print("--- Overall Metrics --- ")
        pprint.pprint(f"Win: {self.success}/{self.iteration}")
        pprint.pprint(f"Failed: {self.failed}/{self.iteration}")
        pprint.pprint(f"Total Reward: {self.total_reward}")





        
    
        

