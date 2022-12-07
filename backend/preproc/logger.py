from typing import Literal
from enum import Enum
from time import asctime, localtime


class FCOLOR(Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    GRAY = 90


class STYLE(Enum):
    RST = 0
    BOLD = 1
    ITALIC = 3
    UNDL = 4


class _ColoredString:
    def __init__(self, s: str, fcolor: int = None, style: int = None) -> None:
        if style:
            self.s = f"\033[{fcolor};{style}m{s}\033[0m"
        else:
            self.s = f"\033[{fcolor}m{s}\033[0m"

    def __str__(self) -> str:
        return self.s


class Logger:
    '''
    Logger class
    -----
    Logs info with colored strings.

    Args:
        `msg`(`str`): The message to log.
    
    Usage:
        `Logger(msg).log(siglevel)`
    '''

    def __init__(self, msg: str) -> None:
        self.msg = msg

    def _timestr() -> str:
        timestr = asctime(localtime())
        return f'[{_ColoredString(timestr, FCOLOR.GRAY.value).s}]'

    def _siglevel(siglevel: Literal['ok', 'warn', 'error', 'info']) -> str:
        match siglevel:
            case 'ok':
                return f'[{_ColoredString("ok", FCOLOR.GREEN.value).s}]'
            case 'warn':
                return f'[{_ColoredString("warn", FCOLOR.YELLOW.value).s}]'
            case 'error':
                return f'[{_ColoredString("error", FCOLOR.RED.value).s}]'
            case 'info':
                return f'[{_ColoredString("info", FCOLOR.BLUE.value).s}]'

    def log(self, siglevel: Literal['ok', 'warn', 'error', 'info']) -> None:
        '''
        Logs the given info with a siglevel.

        Args:
            `siglevel` (`Literal['ok', 'warn', 'error', 'info']`): Signal level that indicates the importance of the message.
        '''
        logstr = ''
        # time section
        logstr += Logger._timestr()
        logstr += ' '
        # siglevel section
        logstr += Logger._siglevel(siglevel)
        logstr += ' '
        # info section
        logstr += self.msg
        print(logstr)
