from __future__ import annotations

from typing import Literal, Type, TypeVar

T = TypeVar("T", bound="ObjectIOMixin")

SerializerBackend = Literal["pickle", "joblib", "cloudpickle"]


class ObjectIOMixin:
    """
    Persistence for the *Python object* (Index/Annoy alias),
    not the raw Annoy index file.

    This relies on __reduce__ from PickleMixin, so it stays stable
    across pickle/joblib/cloudpickle.
    """

    def dump_binary(self, path: str, *, backend: SerializerBackend = "pickle") -> None:
        if backend == "pickle":
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

        if backend == "cloudpickle":
            import cloudpickle
            with open(path, "wb") as f:
                cloudpickle.dump(self, f)
            return

        if backend == "joblib":
            import joblib
            joblib.dump(self, path)
            return

        raise ValueError("backend must be 'pickle', 'joblib', or 'cloudpickle'")

    def save_to_file(self, path: str) -> None:
        self.dump_binary(path, backend="joblib")

    @classmethod
    def load_binary(cls: Type[T], path: str, *, backend: SerializerBackend = "pickle") -> T:
        if backend == "pickle":
            import pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)
        elif backend == "cloudpickle":
            import cloudpickle
            with open(path, "rb") as f:
                obj = cloudpickle.load(f)
        elif backend == "joblib":
            import joblib
            obj = joblib.load(path)
        else:
            raise ValueError("backend must be 'pickle', 'joblib', or 'cloudpickle'")

        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj

    @classmethod
    def load_from_file(cls: Type[T], path: str) -> T:
        return cls.load_binary(path, backend="joblib")
