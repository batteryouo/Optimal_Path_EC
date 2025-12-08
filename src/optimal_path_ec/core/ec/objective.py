import inspect
import math

import numpy as np

class MultiObjective:
    def __init__(self, objectives_func_list, **kwargs):
        # if len(states) != len(objectives_func_list):
        #     raise ValueError(f"len(states) = {len(states)}, but len(objectives_func_list) = {len(objectives_func_list)}")
        self.func_list = []
        self.values = []
        for func in objectives_func_list:
            sig = inspect.signature(func)

            param_names = set(sig.parameters.keys())
            
            has_varkw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            
            self.func_list.append((func, param_names, has_varkw))
                 
        for func, param_names, has_varkw in self.func_list:
            if has_varkw:
                func_kwargs = kwargs
            else:
                common_keys = param_names.intersection(kwargs.keys())
                func_kwargs = {k: kwargs[k] for k in common_keys}
            
            self.values.append(func(**func_kwargs))
        self.states = kwargs
    
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
    
    def __call__(self, **kwargs):
        self.states = kwargs
        self.values = []
        for func, param_names, has_varkw in self.func_list:
            if has_varkw:
                func_kwargs = kwargs
            else:
                common_keys = param_names.intersection(kwargs.keys())
                func_kwargs = {k: kwargs[k] for k in common_keys}
            
            self.values.append(func(**func_kwargs))
        return self.values
