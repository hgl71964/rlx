from abc import ABC, abstractmethod

from typing import Optional, Any, Union


class Node:  # for type annotation
    pass


class Edge(ABC):

    @abstractmethod
    def get_idx(self) -> int:
        pass

    @abstractmethod
    def get_type(self):
        pass

    @abstractmethod
    def set_type(self, type):
        pass

    @abstractmethod
    def get_attr(self) -> Any:
        pass

    @abstractmethod
    def set_attr(self, attr):
        pass

    @abstractmethod
    def get_uses(self) -> list[Node]:
        pass

    @abstractmethod
    def set_uses(self, uses):
        pass

    @abstractmethod
    def get_trace(self) -> Optional[Node]:
        pass

    @staticmethod
    @abstractmethod
    def get_nums_embedding() -> int:
        pass

    @abstractmethod
    def get_embedding(self) -> list[Union[int, float]]:
        pass


class Node(ABC):

    @abstractmethod
    def get_idx(self) -> int:
        pass

    @abstractmethod
    def get_type(self):
        pass

    @abstractmethod
    def set_type(self, type):
        pass

    @abstractmethod
    def get_attr(self) -> Any:
        pass

    @abstractmethod
    def set_attr(self, attr):
        pass

    @abstractmethod
    def get_inputs(self) -> list[Edge]:
        pass

    @abstractmethod
    def get_outputs(self) -> list[Edge]:
        pass

    @staticmethod
    @abstractmethod
    def get_nums_embedding() -> int:
        pass

    @abstractmethod
    def get_embedding(self) -> list[Union[int, float]]:
        pass


class Graph(ABC):

    @abstractmethod
    def get_nodes(self) -> list[Node]:
        pass

    @abstractmethod
    def get_edges(self) -> list[Node]:
        pass
