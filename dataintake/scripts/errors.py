class DuplicateKeyError(Exception):
    """Raised when committing a file would overwrite existing (sub_id, session)
    data and the caller didn't explicitly opt into overwriting it."""
