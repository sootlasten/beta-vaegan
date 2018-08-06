
def overrides(interface_class):
  """Convenience decorator for later overriding the Trainer class."""
  def overrider(method):
    assert(method.__name__ in dir(interface_class))
    return method
  return overrider

