import utility


class Log(object):

    def __init__(self, save_path, name):
        self.save_path, self.name = save_path, name

    def write(self, seq, debug=None):
        if debug is not None:
            utility.write('{}: {}'.format(debug, seq))
        with open('{}/{}'.format(self.save_path, self.name), 'a') as f:
            f.write('{}\n'.format(seq))
