'''Logging utilities.'''

__all__ = ('get_format_kwargs',)

def get_format_kwargs(
        *,
        levelname: bool = True,
        levelno: bool = False,
        name: bool = False,
        asctime: bool = True,
        processName: bool = False,
        process: bool = False,
        threadName: bool = False,
        thread: bool = False,
        pathname: bool = False,
        filename: bool = True,
        funcName: bool = True,
        lineno: bool = True,
) -> dict[str, str]:
    '''Create a dictionary of kwargs to pass to :func:`logging.basicConfig` to
    format log entries the way I like them.

    Arguments are boolean flags indicating the presence or absence of
    :class:`~logging.LogRecord` attributes to display in log entries.

    '''

    prefix_elts: list[str] = []

    if levelname and levelno:
        prefix_elts.append('{levelname}({levelno})')
    elif levelname:
        prefix_elts.append('{levelname}')
    elif levelno:
        prefix_elts.append('LVL{levelno}')

    if name:
        if levelname or levelno:
            prefix_elts[-1] += (':{name}')
        else:
            prefix_elts.append('{name}')

    if asctime:
        prefix_elts.append('{asctime}')

    if processName and process:
        prefix_elts.append('{processName}({process})')
    elif processName:
        prefix_elts.append('{processName}')
    elif process:
        prefix_elts.append('PROC{process}')

    thread_str = None
    if threadName and thread:
        thread_str = '{threadName}({thread})'
    elif threadName:
        thread_str = '{threadName}'
    elif thread:
        thread_str = 'THRD{thread}'

    if thread_str:
        if processName or process:
            prefix_elts[-1] += f':{thread_str}'
        else:
            prefix_elts.append(thread_str)

    if pathname:
        prefix_elts.append('{pathname}')

    if filename:
        if pathname:
            prefix_elts[-1] += ":{filename}"
        else:
            prefix_elts.append('{filename}')

        if lineno:
            prefix_elts[-1] += ":{lineno}"

    if funcName:
        prefix_elts.append('{funcName}')

        if lineno and not (pathname or filename):
            prefix_elts[-1] += ':{lineno}'

    return {
        'style': '{',
        'format': '[{}] {{message}}'.format(' '.join(prefix_elts)),
    }
