from dataclasses import dataclass
from dask.distributed import Client, TimeoutError, LocalCluster
from src.inundation import utils
from typing import Any, Optional
@dataclass
class dask_util():
    address: str = ''
    n_workers: int = 0
    cluster_instance : Optional[Any] = None
        
    def set_workers(self)-> int:
        return utils.specify_max_workers()
    
    def init_cluster(self)->None:
        n = self.set_workers(self)
        client = Client(LocalCluster(n_workers = n, dashboard_address=':8786', processes=False))
        self.address = client.scheduler.addr
        self.n_workers = n 
        print(f"{self.n_workers},{self.address}")
        self.cluster_instance = client
        
    def dask_check(self)->None:
    

        try:
            client = Client(f'{self.address}', timeout='2s')
            #client = Client(n_workers=4,dashboard_address='localhost:8787')
            print(f'connected cluster {self.address}')
        except OSError:
            #print('cluster already running: using default port 8786')
            print(f'could not connect to cluster {self.address}')
            self.init_cluster(self)
        
    def close(self)->None:
        try:
            client = Client(f'{self.address}', timeout='2s')
            #client = Client(n_workers=4,dashboard_address='localhost:8787')
            client.shutdown()
        except OSError:
            #print('cluster already running: using default port 8786')
            #print(f'could not connect to cluster {self.address}')
            #self.init_cluster(self)
            print('no cluster to be closed')