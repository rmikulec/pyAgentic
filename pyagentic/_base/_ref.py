from pydantic import BaseModel

class RefNode(BaseModel):
    """Represents a lazy dotted reference like ref.parent.conversation.goals."""

    def __init__(self, path):
        self._path = path

    def __getattr__(self, key):
        return RefNode(self._path + [key])

    def __call__(self, agent):
        return self.resolve(agent)

    def __repr__(self):
        return f"Ref({'.'.join(self._path)})"

    def resolve(self, agent_reference: dict):
        target = agent_reference
        for part in self._path:
            # skip virtual roots like "self"
            if part in ("self", "agent", "root"):
                continue
            target = target[part]
        return target


class _RefRoot:
    def __getattr__(self, key):
        # default to ref.self.*
        return RefNode([key])


ref = _RefRoot()