


class Print():
    def __init__(self, log_path='log.txt'):
        # Output log file
        self.log_path = log_path
        self.output_log = open(log_path, 'w')
        self.output_log.close()
    
    def __call__(self, *args):
        self.output_log = open(self.log_path, 'a')
        print(*args, file=self.output_log)
        print(*args)
        self.output_log.close()
