import functools
import os
from PIL import Image
from threading import Thread


class TimeoutException(Exception):
    """Customized TimeoutException"""
    pass


def file_does_exist(file_path):
    """
    At this stage, make sure it exists and is a file.
    A directory will be support later.
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)


def is_valid_image(file_path):
    """
    Check if the file is a valid image file.
    """
    if not file_path.lower().endswith((".png", ".jpg", ".jepg", ".gif", ".bmp")):
        return False

    try:
        with Image.open(file_path) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False


def timeout(timeout):
    """
    A timeout deco for detect timeout error in Gemini call as
    Gemini does not have native timeout support.
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [TimeoutException("It took too long, timeout error.")]

            def new_func():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=new_func)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('Error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco
