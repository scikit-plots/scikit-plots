from __future__ import annotations

from typing import Literal, TypeVar

from typing_extensions import Self

T = TypeVar("T", bound="ObjectIOMixin")

SerializerBackend = Literal["pickle", "joblib", "cloudpickle"]


class ObjectIOMixin:
    """
    Persistence for the *Python object* (Index/Annoy alias), not the raw Annoy index file.

    This relies on __reduce__ from PickleMixin, so it stays stable
    across pickle/joblib/cloudpickle.
    """

    def dump_binary(self, path: str, *, backend: SerializerBackend = "pickle") -> None:
        if backend == "pickle":
            import pickle  # noqa: PLC0415

            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

        if backend == "cloudpickle":
            import cloudpickle  # noqa: PLC0415

            with open(path, "wb") as f:
                cloudpickle.dump(self, f)
            return

        if backend == "joblib":
            import joblib  # noqa: PLC0415

            joblib.dump(self, path)
            return

        raise ValueError("backend must be 'pickle', 'joblib', or 'cloudpickle'")

    def save_to_file(self, path: str) -> None:
        self.dump_binary(path, backend="joblib")

    @classmethod
    def load_binary(cls, path: str, *, backend: SerializerBackend = "pickle") -> Self:
        if backend == "pickle":
            import pickle  # noqa: PLC0415

            with open(path, "rb") as f:
                obj = pickle.load(f)  # noqa: S301
        elif backend == "cloudpickle":
            import cloudpickle  # noqa: PLC0415

            with open(path, "rb") as f:
                obj = cloudpickle.load(f)
        elif backend == "joblib":
            import joblib  # noqa: PLC0415

            obj = joblib.load(path)
        else:
            raise ValueError("backend must be 'pickle', 'joblib', or 'cloudpickle'")

        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj

    @classmethod
    def load_from_file(cls, path: str) -> Self:
        return cls.load_binary(path, backend="joblib")
