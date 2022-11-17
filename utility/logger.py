import sys


class Logger:
    
    def __init__(self, path):
        
        self.console = sys.stdout
        self.file = open(path, 'w')
        self.flush()
    
    def write(self, msg):
        
        self.console.write(msg)
        self.file.write(msg)
        self.flush()
    
    def flush(self):
        
        self.console.flush()
        self.file.flush()
    
    def close(self):
        
        self.file.close()