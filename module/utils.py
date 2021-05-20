import os
import joblib

import logging



ROOT = '/Users/shugo/Desktop/SIGNATE/make_env/root'

INPUT = os.path.join(ROOT, 'input')
OUTPUT = os.path.join(ROOT, 'output')
SUBMISSION = os.path.join(ROOT, 'submission')

EXP_NAME = 'exp000' # notebookの名前を自動で取ってきたい
EXP = os.path.join(OUTPUT, EXP_NAME)
PREDS = os.path.join(EXP, 'preds')
TRAINED = os.path.join(EXP, 'trained')
FEATURE = os.path.join(EXP, 'feature')
REPORTS = os.path.join(EXP, 'reports')

dirs = [
        OUTPUT,
        SUBMISSION,
        FEATURE,
        EXP,
        PREDS,
        TRAINED,
        REPORTS
        ]

for v in dirs:
    if not os.path.isdir(v):
        print(f'making {v}')
        os.makedirs(v, exist_ok=True)
            

class Util:
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)
        
    @classmethod
    def load(cls, path):
        return joblib.load(path)
    

class Logger:
    def __init__(self, path):
        self.general_logger = logging.getLogger(path) # loggerを設定
        stream_handler = logging.StreamHandler() # コンソールへ出力
        file_general_handler = logging.FileHandler(os.path.join(path, 'Experiment.log')) # ファイルへ出力
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(fil_general_handler)
            se,f.general_logger.setLevel(logging.INFO)
            
    def info(self, message):
        self.general_logger.info(f'{self.now_string()}-{message}')
        
    @staticmethod
    def now_string():
        return str(datetime.datetime.now.strftime('%Y-%m-%d %H:%M:%S'))
                                 
        
    def save(self):
        return Logger(REPORTS)