from dataclasses import dataclass
from typing import Sequence
from pocomc import Prior


@dataclass
class BaseModel:
    names: Sequence[str]
    limits: Sequence[Sequence]
    init: Sequence[float]
    prior: Sequence

    @classmethod
    def fromdict(__cls__, _dict):
        names = []
        limits = []
        init = []
        prior = []
        for key, row in _dict.items():
            names.append(key)
            limits.append(row[0])
            init.append(row[1])
            prior.append(row[2])
        return __cls__(names=names, limits=limits, init=init, prior=Prior(prior))
