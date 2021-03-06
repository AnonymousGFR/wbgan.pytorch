import numpy as np
import logging, os
import datetime
import sys

from .plot_utils import plot_figure

FORMAT = "[%(levelname)s]: %(message)s [%(name)s][%(filename)s:%(funcName)s():%(lineno)s][%(asctime)s.%(msecs)03d]"
DATEFMT = '%Y/%m/%d %H:%M:%S'


def logging_init(filename=None, level=logging.INFO, correct_time=False):
  def beijing(sec, what):
    '''sec and what is unused.'''
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

  if correct_time:
    logging.Formatter.converter = beijing

  logging.basicConfig(level=level,
                      format=FORMAT,
                      datefmt=DATEFMT,
                      filename=None, filemode='w')
  logger = logging.getLogger()

  # consoleHandler = logging.StreamHandler()
  # logger.addHandler(consoleHandler)

  if filename:
    logger_handler = logging.FileHandler(filename=filename, mode='w')
    logger_handler.setLevel(level=level)
    logger_handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))
    logger.addHandler(logger_handler)

  def info_msg(*argv):
    # remove formats
    org_formatters = []
    for handler in logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    logger.info(*argv)

    # restore formats
    for handler, formatter in zip(logger.handlers, org_formatters):
      handler.setFormatter(formatter)

  logger.info_msg = info_msg
  return logger

  # logging.error('There are something wrong', exc_info=True)


def get_root_logger(filename, stream=True, level=logging.INFO):
  logger = logging.getLogger()
  logger.setLevel(level)
  set_hander(logger=logger, filename=filename, stream=stream, level=level)
  return logger


def get_logger(filename, stream=True, level=logging.INFO):
  """

  :param filename:
  :param propagate: whether log to stdout
  :return:
  """
  logger = logging.getLogger(filename)
  logger.setLevel(level)
  set_hander(logger=logger, filename=filename, stream=stream, level=level)
  return logger


def set_hander(logger, filename, stream=True, level=logging.INFO):
  formatter = logging.Formatter(FORMAT, datefmt=DATEFMT)

  file_hander = logging.FileHandler(filename=filename, mode='a')
  file_hander.setLevel(level=level)
  file_hander.setFormatter(formatter)
  logger.addHandler(file_hander)

  def info_msg(*argv):
    # remove formats
    org_formatters = []
    for handler in logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("  %(message)s"))

    logger.info(*argv)

    # restore formats
    for handler, formatter in zip(logger.handlers, org_formatters):
      handler.setFormatter(formatter)

  logger.info_msg = info_msg

  if stream:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

  logger.propagate = False
  return logger


class StreamToLogger(object):
  """
  Fake file-like stream object that redirects writes to a logger instance.
  """

  def __init__(self, logger):
    self.logger = logger
    self.linebuf = ''

  def write(self, buf):
    buf = '> ' + buf
    for line in buf.rstrip().splitlines():
      self.logger.info_msg(line.rstrip())

  def flush(self):
    pass

  def getvalue(self):
    pass

  def close(self):
    pass


def redirect_print_to_logger(logger, ):
  sl = StreamToLogger(logger)
  sys.stdout = sl
  sys.stderr = sl
  pass


class TextLogger(object):
  """
  # Logstyle is either:
  # '%#.#f' for floating point representation in text
  # '%#.#e' for exponent representation in text
  """
  def __init__(self, log_root, reinitialize=False, logstyle='%3.3f'):
    self.root = log_root
    if not os.path.exists(self.root):
      os.mkdir(self.root)
    self.reinitialize = reinitialize
    self.metrics = []
    # One of '%3.3f' or like '%3.3e'
    self.logstyle = logstyle

  def reinit(self, item):
    """
      Delete log if re-starting and log already exists
    """
    if os.path.exists('%s/%s.log' % (self.root, item)):
      os.remove('%s/%s.log' % (self.root, item))


  def log(self, itr, **kwargs):
    """
    Log in plaintext;
    """
    for arg in kwargs:
      if arg not in self.metrics:
        if self.reinitialize:
          self.reinit(arg)
        self.metrics += [arg]
      with open('%s/%s.log' % (self.root, arg), 'a') as f:
        f.write('%3d: %s\n' % (itr, self.logstyle % kwargs[arg]))

  def log_axes(self, **kwargs):
    names = []
    filepaths = []
    for arg in kwargs:
      filename = '%s/%s.log' % (self.root, arg)
      names.append(arg)
      filepaths.append(filename)

    plot_figure(names=names, filepaths=filepaths,
                outdir=self.root, in_one_axes=False)

  def log_ax(self, **kwargs):
    names = []
    filepaths = []
    for arg in kwargs:
      filename = '%s/%s.log' % (self.root, arg)
      names.append(arg)
      filepaths.append(filename)

    plot_figure(names=names, filepaths=filepaths,
                outdir=self.root, in_one_axes=True)

  def logstr(self, itr, **kwargs):
    for arg in kwargs:
      if arg not in self.metrics:
        if self.reinitialize:
          self.reinit(arg)
        self.metrics += [arg]
      with open('%s/%s.log' % (self.root, arg), 'a') as f:
        f.write('%3d: %s\n' % (itr, kwargs[arg]))


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
