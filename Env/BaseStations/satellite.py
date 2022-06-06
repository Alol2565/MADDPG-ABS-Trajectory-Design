from nodes import SP_Node

class Satellite(SP_Node):
    def __init__(self, id, power, initial_location=...):
        super().__init__(id, power, initial_location)

    def reset(self):
        return super().reset()