import threading

class Thread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        # variable to store the output of the function after execution
        self._return = None
        # variable to store any errors that may occur during execution and reraise them
        self.exc = None
        
    def run(self):
        try:
            if self._target is not None:
                self._return = self._target(*self._args,**self._kwargs)
        except BaseException as e:self.exc = e #store any errors that occur to variable
        
    def join(self):
        threading.Thread.join(self)
        if self.exc:raise self.exc
        return self._return
