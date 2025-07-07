class Pipline():
    def __init__(self, instance_name,duration):
        self.instance_name = instance_name
        self.state = False
        self.activate_state = False
        self.duration = duration
    def activate(self, start):
        self.start = start
        self.end = start + self.duration 
        self.activate_state = True
        return self.end  # 返回准备完成时间
    
    def state_check(self, time):
        if time < self.end:
            self.state = False
        else:
            self.state = True
        return self.state
    def reset(self):
        self.activate_state = False
        self.state = False