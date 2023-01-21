class ProgressBar:
    def __init__(self, message: str, target: int) -> None:
        self.message = message.replace('%target%', str(target))
        self.target = target

    def __del__(self):
        self.message = None
        self.target = None

    def update(self, new_value: int) -> None:
        message = self.message.replace('%current%', str(new_value))

        progress = f'{new_value / self.target * 100:.2f}%'
        message = message.replace('%progress%', progress)

        print(message, end='\r')

    def close(self):
        print('')  # New line
        self.__del__()
