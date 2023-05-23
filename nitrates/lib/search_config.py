import json

class Config:
    def __init__(self, filepath):
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)

        self.ERRORS = data.get('ERRORS', [])
        self.WARNINGS = data.get('WARNINGS', [])
        self.queueID = data.get('queueID', None)
        self.triggerID = data.get('triggerID', None)
        self.trigtime = data.get('trigtime', None)
        self.workdir = False

        config = data.get('config', {})
        for key in config:
            setattr(self, key, config[key])
