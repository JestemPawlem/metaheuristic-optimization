COLOR_INFO = '\033[1;95m'
COLOR_ERROR = '\033[1;91m'
COLOR_RESET = '\033[0m'


def print_info(message: str, **kwargs) -> None:
    print(f'{COLOR_INFO}{message}{COLOR_RESET}', **kwargs)

def print_error(message: str, **kwargs) -> None:
    print(f'{COLOR_ERROR}{message}{COLOR_RESET}', **kwargs)