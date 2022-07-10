


class Print():
    def __init__(self) -> None:
        # Output log file
        self.output_log = open('log.txt', 'w')
        self.output_log.close()
    
    def __call__(self, *args):
        self.output_log = open('log.txt', 'a')
        print(*args, file=self.output_log)
        print(*args)
        self.output_log.close()