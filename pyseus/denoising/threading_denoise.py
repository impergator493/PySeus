import threading

class ThreadingDenoised(threading.Thread):
    def __init__(self, data_obj, function, finished_cb, args):
        threading.Thread.__init__(self)
        self.data_obj = data_obj
        self.function = function
        self.args = args
        self.finished_cb = finished_cb

    def run(self):
        self.data_obj = self.function(*self.args)
        self.finished_cb(self.data_obj)

