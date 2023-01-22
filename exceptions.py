class InvalidGraphException(Exception):
    """Raised when a graph is initialized with improper parameters."""


class InvalidNodeException(Exception):
    """Raised when a method is called on a node that doesn't exist."""


class InvalidMessageException(Exception):
    """Raised when a ProgressBar is initialized with an improper message."""
