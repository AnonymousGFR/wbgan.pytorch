import os
import torch
import torchvision
import torch.nn as nn


def sample_sheet(G, classes_per_sheet, num_classes, samples_per_class, parallel, z_=None):
  image_filenames = []
  images = []
  # loop over total number of sheets
  for i in range(num_classes // classes_per_sheet):
    ims = []
    y = torch.arange(i * classes_per_sheet, (i + 1) * classes_per_sheet, device='cuda')
    for j in range(samples_per_class):
      if (z_ is not None) and hasattr(z_, 'sample_') and classes_per_sheet <= z_.size(0):
        z_.sample_()
      else:
        z_ = torch.randn(classes_per_sheet, G.dim_z, device='cuda')
      with torch.no_grad():
        if parallel:
          o = nn.parallel.data_parallel(G, (z_[:classes_per_sheet], G.shared(y)))
        else:
          o = G(z_[:classes_per_sheet], G.shared(y))

      ims += [o.data.cpu()]
    # This line should properly unroll the images
    out_ims = torch.stack(ims, 1).view(-1, ims[0].shape[1], ims[0].shape[2],
                                       ims[0].shape[3]).data.float().cpu()
    # The path for the samples
    image_filename = 'samples/%d.jpg' % (i)
    image = torchvision.utils.make_grid(out_ims, nrow=samples_per_class, normalize=True)
    image = image.view(1, *image.shape)
    image_filenames.append(image_filename)
    images.append(image)

  return image_filenames, images


# Interp function; expects x0 and x1 to be of shape (shape0, 1, rest_of_shape..)
def interp(x0, x1, num_midpoints):
  lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cuda').to(x0.dtype)
  return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))

# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device='cuda'):
  return torch.randint(low=0, high=num_classes, size=(batch_size,),
          device=device, dtype=torch.int64, requires_grad=False)


def interp_sheet(G, num_per_sheet, num_midpoints, num_classes, parallel,
                 sheet_number=0, fix_z=False, fix_y=False, device='cuda'):
  """
  # interp sheet function
  # Supports full, class-wise and intra-class interpolation
  :param G:
  :param num_per_sheet:
  :param num_midpoints:
  :param num_classes:
  :param parallel:
  :param sheet_number:
  :param fix_z:
  :param fix_y:
  :param device:
  :return:
  """
  # Prepare zs and ys
  # If fix Z, only sample 1 z per row
  if fix_z:
    zs = torch.randn(num_per_sheet, 1, G.dim_z, device=device)
    zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, G.dim_z)
  else:
    zs = interp(torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                num_midpoints).view(-1, G.dim_z)
  # If fix y, only sample 1 y per row
  if fix_y:
    ys = sample_1hot(num_per_sheet, num_classes)
    ys = G.shared(ys).view(num_per_sheet, 1, -1)
    ys = ys.repeat(1, num_midpoints + 2, 1).view(num_per_sheet * (num_midpoints + 2), -1)
  else:
    ys = interp(G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                num_midpoints).view(num_per_sheet * (num_midpoints + 2), -1)
  # Run the net--note that we've already passed y through G.shared.
  with torch.no_grad():
    if parallel:
      out_ims = nn.parallel.data_parallel(G, (zs, ys)).data.cpu()
    else:
      out_ims = G(zs, ys).data.cpu()
  interp_style = '' + ('Z' if not fix_z else '') + ('Y' if not fix_y else '')
  image_filename = 'interp/%s_%d.jpg' % (interp_style, sheet_number)
  image = torchvision.utils.make_grid(out_ims, nrow=num_midpoints + 2, normalize=True)
  image = image.view(1, *image.shape)
  return image_filename, image
