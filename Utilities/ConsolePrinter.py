class ConsolePrinter:

    def __init__(self, header_list = None):
        self.header_list = header_list
        if header_list is not None:
            self.PrintHeader()
 
    def PrintHeader(self):
        num_dash = 16*len(self.header_list) + 1
        print('{}'.format('-'*num_dash))
        line = '|'
        for name in self.header_list:
            line = line + ' {:^13} |'.format(name)
        print(line)
        print('{}'.format('-'*num_dash))

    def PrintContent(self, value):
        if self.header_list is None:
            self.header_list = value.keys()
            self.PrintHeader()

        line = '|'
        for key in self.header_list:
            val = value[key]
            try:
                val = float(val) 
                line = line + ' {:^13.3f} |'.format(val)
            except ValueError:
                line = line + ' {:^13} |'.format(val)
        print(line)
