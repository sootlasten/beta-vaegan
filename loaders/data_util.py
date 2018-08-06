from abc import ABC, abstractmethod


class DataUtil(ABC):
  @abstractmethod
  def get_trainloader(self, batch_size):
    pass
  
  @property
  @abstractmethod
  def testdata(self):
    pass

