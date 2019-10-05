import matplotlib.pyplot as plt
import os
import pprint
from collections import OrderedDict

from template_lib.utils import plot_utils
from template_lib.utils.parse_tensorboard import SummaryReader

def parse_tensorboard(args, myargs):
  config = getattr(myargs.config, args.command)
  print(pprint.pformat(OrderedDict(config)))
  data_dict = {}

  for label, line in config.lines.items():
    print("Parsing: %s"%line.tbdir)
    summary_reader = SummaryReader(tbdir=line.tbdir)
    tags = summary_reader.get_tags()
    print(pprint.pformat(tags))
    data = summary_reader.get_scalar(tag=config.tag, use_dump=config.use_dump)
    data_dict[label] = data

  matplot = plot_utils.MatPlot()
  plt.style.use('classic')
  fig, ax = matplot.get_fig_and_ax()
  for label, line in config.lines.items():
    data = data_dict[label]
    ax.plot(data[0], data[1], **{"label": label,
                                 **getattr(line, 'property', {})})
  ax.legend(loc='best')
  ax.set_ylim(config.ylim)
  ax.set_xlim(config.xlim)
  fontdict = {'fontsize': 17}
  ax.set_xlabel('Epoch', fontdict=fontdict)
  ax.set_ylabel('FID', fontdict)
  ax.set_title(config.title, fontdict=fontdict)

  fig_name = config.tag.replace('/', '_') + '.png'
  filepath = os.path.join(args.outdir, fig_name)
  matplot.save_to_png(fig=fig, filepath=filepath)

  fig_name = config.tag.replace('/', '_') + '.pdf'
  filepath = os.path.join(args.outdir, fig_name)
  matplot.save_to_pdf(fig=fig, filepath=filepath)

  pass