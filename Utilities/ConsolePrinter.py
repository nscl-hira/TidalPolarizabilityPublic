import numpy as np
import sys
from tqdm import tqdm
from mpi4py import MPI
import time

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

tags = enum('Increment', 'Nothing', 'Finished')

def polling_receive(comm, duration):
    #print('polling_receive %d', comm.Get_rank())
    # Set this to 0 for maximum responsiveness, but that will peg CPU to 100%
    sleep_seconds = duration/100
    start = time.time()
    received = False
    while time.time() - start < duration and not received:
        if sleep_seconds > 0:
           received = comm.Iprobe(source=MPI.ANY_SOURCE)
           time.sleep(sleep_seconds)
           #print(time.time() - start)
    if not received:
        return None, tags.Nothing, None 

    status = MPI.Status()
    result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

    return status.Get_source(), status.Get_tag(), result

class Console:

    def __init__(self, comm, total):
        self.comm = comm
        if comm is not None:
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
            recvdata = np.array(0, 'i')
            senddata = np.array(total, 'i')
                
            comm.Reduce([senddata, 1, MPI.INT], [recvdata, 1, MPI.INT], op=MPI.SUM, root=0)
            if self.rank == 0:
                self.total = recvdata
        else:
            self.size = 1
            self.rank = 0
        self.nworkers = self.size
        

    def PrintContent(self, value):
        pass
  
    def PrintError(self, value):
        pass

    def Close(self):
        if self.rank == 0 and self.comm is not None:
            self.nworkers = self.nworkers - 1
            while self.nworkers > 0:
                status = MPI.Status()
                value = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == tags.Increment:
                    self.PrintContent(value)
                elif tag == tags.Finished:
                    self.nworkers = self.nworkers - 1
              
        elif self.rank > 0:
            req = self.comm.isend(None, tag=tags.Finished, dest=0)
            req.wait()
        self.comm.Barrier()

    def ListenFor(self, duration=0.1):
        if self.rank == 0 and self.comm is not None and self.nworkers > 1:
            start = time.time()
            while time.time() - start < duration:
                source, tag, value = polling_receive(self.comm, duration)
                if tag==tags.Increment:
                    self.PrintContent(value)
                elif tag==tags.Finished:
                    self.nworkers = self.nworkers - 1




class ConsolePrinter(Console):

    def __init__(self, header_list=None, total=0, comm=None, **kwargs):
        super().__init__(comm, total)

        self.header_list = header_list
        if header_list is not None and self.rank == 0:
            self._PrintHeader()
 
    def _PrintHeader(self):
        num_dash = 16*len(self.header_list) + 1
        print('{}'.format('-'*num_dash))
        line = '|'
        for name in self.header_list:
            line = line + ' {:^13} |'.format(name)
        print(line)
        print('{}'.format('-'*num_dash))

    def PrintContent(self, value):
        if self.rank == 0:
            if self.header_list is None:
                self.header_list = value.keys()
                self._PrintHeader()

            line = '|'
            for key in self.header_list:
                val = value[key]
                try:
                    val = float(val) 
                    line = line + ' {:^13.3f} |'.format(val)
                except ValueError:
                    line = line + ' {:^13} |'.format(val)
            print(line)
            sys.stdout.flush()
        else:
            req = self.comm.isend(value, tag=tags.Increment, dest=0)
            req.wait()

    def PrintError(self, value):
        pass



class ConsolePBar(Console):

    def __init__(self, header_list=None, total=0, comm=None, **kwargs):
        super().__init__(comm, total)
        disable = True

        if self.rank == 0:
            self.pbar = tqdm(total=self.total, ncols=100, smoothing=0)

    def PrintContent(self, value):
        if self.rank == 0:
            self.pbar.update(1)
            self.pbar.refresh()
        else:
            req = self.comm.isend(value, tag=tags.Increment, dest=0)
            req.wait()   

    def PrintError(self, value):
        self.PrintContent(value)

    def Close(self):
        super().Close()
        if self.rank == 0:
            self.pbar.close()
            print('')


