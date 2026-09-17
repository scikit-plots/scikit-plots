"""
A named, pinned set of corpora.

What a collection owns that a corpus does not is **membership**, and the
generation each member is pinned to. That pin is the whole design.

A collection that named "whatever is current in that directory" would mean
different things when read twice, which is precisely the failure the generation
contract removes one level down: an artifact is an immutable generation and a
pointer says which one is current, so a reader holding a generation keeps it.
A collection binds to a generation for the same reason, and the retention the
artifact already provides is what makes the pin meaningful rather than merely
recorded.

Notes
-----
**User.** ``add`` returns a new collection rather than mutating this one, so a
collection you are holding cannot change underneath you. Pin a member by adding
it; re-add it to move the pin forward deliberately.

**Developer.** Identity is derived through the canonical encoder, over the
member names and their pinned generations, so it does not depend on the order
members were added. A collection is published through the same
candidate-verify-commit shape as the artifact it contains; this is the fourth
place that shape appears in the package, which is the recorded cost of each
submodule standing alone.

See Also
--------
scikitplot.corpus._artifact.ANNIndexArtifact : The generation contract a member is pinned to.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shutil
import uuid

from ._artifact import ANNIndexArtifact
from ._canonical import canonical_digest

__all__ = ["CollectionError", "CorpusCollection", "Member"]

_POINTER_NAME = "current.json"
_SNAPSHOT_NAME = "collection.json"
_SCHEMA = "collection/1"
_RETAINED_GENERATIONS = 2


class CollectionError(ValueError):
    """Raised when a collection cannot be built, read or published."""


@dataclasses.dataclass(frozen=True)
class Member:
    """
    One corpus in a collection, pinned to a generation.

    Parameters
    ----------
    name : str
        Name the collection knows this corpus by.
    path : str
        Artifact root, as supplied.
    generation : str
        Fingerprint of the generation this member is pinned to.
    generation_path : str
        The generation directory itself, which is what gets opened. Recorded so
        a pin survives the corpus publishing a replacement.
    """

    name: str
    path: str
    generation: str
    generation_path: str


@dataclasses.dataclass(frozen=True)
class CorpusCollection:
    """
    An immutable, named set of corpora pinned to specific generations.

    Parameters
    ----------
    name : str
        Collection name.
    members : tuple of Member, optional
        Members, in the order added. Order does not affect identity.
    """

    name: str
    members: tuple[Member, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Tuple of str: Member names, in the order added."""
        return tuple(member.name for member in self.members)

    @property
    def identity(self) -> str:
        """
        Full SHA-256 digest of the name and the pinned membership.

        Notes
        -----
        **Developer.** Derived from a mapping, which the canonical encoder
        orders, so two collections built by different call orders are one
        collection. Paths are deliberately excluded: the same build reached
        through a different path is the same evidence.
        """
        return canonical_digest(
            {
                "schema": _SCHEMA,
                "name": self.name,
                "members": {m.name: m.generation for m in self.members},
            }
        )

    def add(self, name: str, path: str | pathlib.Path) -> CorpusCollection:
        """
        Return a new collection with ``path`` pinned under ``name``.

        Parameters
        ----------
        name : str
            Name to know this corpus by.
        path : str or pathlib.Path
            Artifact root. Its current generation is what gets pinned.

        Returns
        -------
        CorpusCollection
            A new collection. This one is unchanged.

        Raises
        ------
        CollectionError
            If ``name`` is already used, or the artifact cannot be opened. Both
            are refused here, where the mistake was made, rather than at the
            first query -- by then the offending call is long gone.
        """
        if name in self.names:
            raise CollectionError(
                f"{name!r} is already a member of collection {self.name!r}; "
                "one name cannot mean two corpora."
            )
        root = pathlib.Path(os.fspath(path))
        try:
            artifact = ANNIndexArtifact.open(root)
        except Exception as exc:  # noqa: BLE001 - reported with its cause
            raise CollectionError(f"cannot add {name!r} from {root}: {exc}") from exc
        member = Member(
            name=name,
            path=str(root),
            generation=artifact.generation.fingerprint,
            generation_path=str(artifact.path),
        )
        return dataclasses.replace(self, members=(*self.members, member))

    def member(self, name: str) -> Member:
        """
        Return the member named ``name``.

        Raises
        ------
        CollectionError
            Naming what was asked for and what the collection holds.
        """
        for member in self.members:
            if member.name == name:
                return member
        raise CollectionError(
            f"{name!r} is not a member of collection {self.name!r}; "
            f"it holds {list(self.names)}."
        )

    def open(self, name: str) -> ANNIndexArtifact:
        """
        Open a member at the generation it was pinned to.

        Notes
        -----
        **User.** This opens the pinned generation, not whatever the corpus has
        published since. If the pinned generation has been pruned, the failure
        says so rather than silently returning a newer build.
        """
        member = self.member(name)
        generation = pathlib.Path(member.generation_path)
        if not generation.is_dir():
            raise CollectionError(
                f"member {name!r} is pinned to generation {member.generation}, "
                f"which is no longer present at {generation}. It may have been "
                "pruned; re-add the member to pin the current generation."
            )
        return ANNIndexArtifact.open(generation)

    def save(self, directory: str | pathlib.Path) -> pathlib.Path:
        """
        Publish this collection as a generation behind a pointer.

        Returns
        -------
        pathlib.Path
            The generation directory written.

        Raises
        ------
        OSError
            If the snapshot could not be written. Nothing is published then, and
            any previously published snapshot is untouched.
        """
        root = pathlib.Path(os.fspath(directory))
        root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "schema": _SCHEMA,
                "name": self.name,
                "identity": self.identity,
                "members": [dataclasses.asdict(m) for m in self.members],
            },
            indent=2,
            sort_keys=True,
        )
        generation = root / f"generation-{self.identity}"
        if not generation.is_dir():
            candidate = root / f".candidate-{uuid.uuid4().hex}"
            try:
                candidate.mkdir(parents=True)
                (candidate / _SNAPSHOT_NAME).write_text(payload, encoding="utf-8")
                # Verify by reloading, so the pointer never names a snapshot
                # nobody has read.
                type(self).load(candidate)
                os.replace(candidate, generation)
            except BaseException:
                shutil.rmtree(candidate, ignore_errors=True)
                raise
        (root / _POINTER_NAME).write_text(
            json.dumps({"generation": generation.name}, indent=2), encoding="utf-8"
        )
        stale = sorted(
            (
                child
                for child in root.iterdir()
                if child.is_dir()
                and child.name.startswith("generation-")
                and child.name != generation.name
            ),
            key=lambda child: child.stat().st_mtime,
            reverse=True,
        )
        for child in stale[_RETAINED_GENERATIONS - 1 :]:
            shutil.rmtree(child, ignore_errors=True)
        return generation

    @classmethod
    def load(cls, directory: str | pathlib.Path) -> CorpusCollection:
        """
        Load a published collection, or a snapshot directory directly.

        Raises
        ------
        CollectionError
            If the root has no readable pointer, or the snapshot was written
            under a different schema.
        """
        root = pathlib.Path(os.fspath(directory))
        if not (root / _SNAPSHOT_NAME).is_file():
            pointer = root / _POINTER_NAME
            if not pointer.is_file():
                raise CollectionError(
                    f"{root} has no {_POINTER_NAME}; there is no way to know "
                    "which snapshot is current."
                )
            root = root / json.loads(pointer.read_text(encoding="utf-8"))["generation"]
        state = json.loads((root / _SNAPSHOT_NAME).read_text(encoding="utf-8"))
        if state.get("schema") != _SCHEMA:
            raise CollectionError(
                f"snapshot at {root} was written under schema "
                f"{state.get('schema')!r}, not {_SCHEMA!r}."
            )
        return cls(
            name=state["name"],
            members=tuple(Member(**entry) for entry in state["members"]),
        )
