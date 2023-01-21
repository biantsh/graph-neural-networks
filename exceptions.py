class InvalidGraphException(Exception):
    """Used when a graph is initialized with improper parameters."""


class InvalidNodeException(Exception):
    """Used when a method is called on a node that doesn't exist."""


class InvalidMessageException(Exception):
    """Used when a ProgressBar is initialized with an improper message."""
