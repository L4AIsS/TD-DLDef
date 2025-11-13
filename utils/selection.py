from typing import List
from utils.Thompson import ThompsonSampling


class Selector(object):

    class Element(object):
        def __init__(self, name: str, selected: int = 0):
            self.name = name
            self.selected = selected

        def record(self):
            self.selected += 1

        @property
        def score(self):
            return 1.0 / (self.selected + 1)

    def __init__(self, layer_types: List[str], layer_conditions: dict):
        self.__pool = {name: self.Element(name=name) for name in layer_types}
        self.__layer_conditions = layer_conditions
        self.__thompson_sampling = ThompsonSampling(layer_types, 1, 1) # 论文建议

    def update(self, name):
        if name == 'input_object':
            return
        self.__thompson_sampling.update(name, self.__thompson_sampling.is_reward(name))
        return

    def coverage(self):
        return self.__thompson_sampling.cal_coverage()

    def branch_lines(self):
        return self.__thompson_sampling.cal_branch_and_lines()

    def get_single_record(self):
        return self.__thompson_sampling.get_single_record()

    def choose_element(self, pool: List[str], **kwargs):
        # 挑选候选者
        candidates = []
        for el_name in pool:
            cond = self.__layer_conditions.get(el_name, None)
            if cond is None or cond(**kwargs):  # available 的 layer
                candidates.append(self.__pool[el_name].name)

        candidate = self.__thompson_sampling.choose_arm(candidates)
        return candidate
        # return random.choice(candidates).name

