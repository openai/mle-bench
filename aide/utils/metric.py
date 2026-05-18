from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any

import numpy as np
from dataclasses_json import DataClassJsonMixin


@dataclass
@total_ordering
class MetricValue(DataClassJsonMixin):
    """
    Represents the value of a metric to be optimized, which can be compared to other metric values.
    Comparisons (and max, min) are based on which value is better, not which is larger.
    """

    value: float | int | np.number | np.floating | np.ndarray | None
    maximize: bool | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.value is not None:#父节点的metric有值
            assert isinstance(self.value, (float, int, np.number, np.floating))
            self.value = float(self.value)

    def __gt__(self, other) -> bool:
        """True if self is a _better_ (not necessarily larger) metric value than other"""
        if self.value is None:
            return False
        if other.value is None:
            return True

#self和otheer的value相等则返回False，强制要求优化方向一致
        assert type(self) is type(other) and (self.maximize == other.maximize)
        if self.value == other.value:
            return False

# 根据优化方向决定比较结果
        comp = self.value > other.value
        return comp if self.maximize else not comp  # type: ignore
    
    def __lt__(self, other):
        return not self.__gt__(other)

    def __eq__(self, other: Any) -> bool:
        return self.value == other.value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.maximize is None:
            opt_dir = "?"
        elif self.maximize:
            opt_dir = "↑"
        else:
            opt_dir = "↓"
        return f"Metric{opt_dir}({self.value_npsafe:.4f})"

    @property
    def is_worst(self):
        """True if the metric value is the worst possible value."""
        return self.value is None

    @property
    def value_npsafe(self):
        return self.value if self.value is not None else float("nan")


@dataclass
class WorstMetricValue(MetricValue):
    """
    Represents an invalid metric value, e.g. when the agent creates a buggy solution.
    Always compares worse than any valid metric value.
    """

    value: None = None

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
