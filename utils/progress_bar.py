from exceptions import InvalidMessageException


class ProgressBar:
    """Class for printing current progress on tasks with many iterations.

    Usage: the `message` parameter should contain the strings '%target%',
    '%current%' (and optionally '%progress%'), which will be replaced with
    their respective values as the progress bar is updated.
    """
    def __init__(self, message: str, target: int) -> None:
        if not ('%target%' in message and '%current%' in message):
            raise InvalidMessageException(
                'The `message` parameter should contain the strings '
                '\'%target%\', \'%current%\'.'
            )

        self.message = message.replace('%target%', str(target))
        self.target = target

    def __del__(self):
        self.message = None
        self.target = None

    def update(self, current_value: int) -> None:
        """Print a message for the current value received."""
        message = self.message.replace('%current%', str(current_value))

        if '%progress%' in message:
            progress = f'{current_value / self.target * 100:.2f}%'
            message = message.replace('%progress%', progress)

        print(message, end='\r')

    def close(self):
        """Close the progress bar upon finishing or cancellation."""
        print('')  # Prevents future text from overwriting progress bar
        self.__del__()
