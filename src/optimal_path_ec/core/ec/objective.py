import math

import numpy as np

class MultiObjective:
    def __init__(self, states, objectives_func_list):
        
        if len(states) != len(objectives_func_list):
            raise ValueError(f"len(states) = {len(states)}, but len(objectives_func_list) = {len(objectives_func_list)}")
        
        self.states = list(states)
        self.func_list = list(objectives_func_list)
        
        self.values = [func(state) for func, state in zip(self.func_list, self.states)]
    
    def compareValue(self, other, compare_list):
        results = []
        if len(self.values) != len(compare_list):
            raise ValueError("compare_list length must match number of objectives")
        valid = {"min", "max"}
        if not all(c in valid for c in compare_list):
            raise ValueError("compare_list items must be 'min' or 'max'")

        for value, other_value, logic in zip(self, other, compare_list):

            if logic == "min":
                results.append(value < other_value)

            else:  # logic == "max"
                results.append(value > other_value)

        return results

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]
    
    def __call__(self, states):
        self.states = states
        self.values = [func(state) for func, state in zip(self.func_list, states)] 
        return self.values
