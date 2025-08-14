# pylint: disable=missing-function-docstring

"""file_info."""

from .._entities._skplt_object import _ScikitplotObject


class FileInfo(_ScikitplotObject):
    """
    Metadata about a file or directory.
    """

    def __init__(self, path, is_dir, file_size):
        self._path = path
        self._is_dir = is_dir
        self._bytes = file_size

    def __eq__(self, other):  # noqa: D105
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def path(self):
        """String path of the file or directory."""
        return self._path

    @property
    def is_dir(self):
        """Whether the FileInfo corresponds to a directory."""
        return self._is_dir

    @property
    def file_size(self):
        """Size of the file or directory. If the FileInfo is a directory, returns None."""
        return self._bytes

    @classmethod
    def from_proto(cls, proto):  # noqa: D102
        # return cls(proto.path, proto.is_dir, proto.file_size)
        pass

    # def to_proto(self):  # noqa: D102
    #     # from mlflow.protos.service_pb2 import FileInfo as ProtoFileInfo
    #     proto = ProtoFileInfo()
    #     proto.path = self.path
    #     proto.is_dir = self.is_dir
    #     if self.file_size:
    #         proto.file_size = self.file_size
    #     return proto
