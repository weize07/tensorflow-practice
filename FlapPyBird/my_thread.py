import threading

class MyThread(threading.Thread):
    def __init__(self, engine):
        self.engine = engine
        threading.Thread.__init__(self)

    def run(self):
        self.engine.start()
