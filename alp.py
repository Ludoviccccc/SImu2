class alp:
    def __init__(self):
        pass
    def reward(self,point, goal):
        reward = - self.loss(goal,point)
        return reward
    def loss(self,goal:np.ndarray, elements:np.ndarray):
        if type(goal)!=float:
            a = goal.reshape(-1,1)
        else:
            a = np.array([goal]).reshape(-1,1)
        out = np.sum((a -elements)**2,axis=0)
        return out
    def alp(self,history_points,
