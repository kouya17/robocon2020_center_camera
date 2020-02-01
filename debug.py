import pathlib
import logging
import logging.config

abs_dir = str(pathlib.Path(__file__).parent.resolve())
logging.config.fileConfig(abs_dir + '/logging.conf')

# create logger
_logger = logging.getLogger('develop')


def logger():
    return _logger


if __name__ == '__main__':
    _logger.debug('debug message')
    _logger.info('info message')
    _logger.warning('warn message')
    _logger.error('error message')
    _logger.critical('critical message')