class LinearSchedule(object):
    def __init__(self, timestep=1e6, final=0.95, initial=0):
        self.timestep = timestep
        self.final = final
        self.initial = initial
        
    def value(self, time):
        return self.initial + min(time / self.timestep, 1.0) * (self.final - self.initial)