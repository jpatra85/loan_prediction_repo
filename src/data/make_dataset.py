# -*- coding: utf-8 -*-
#import preprocess_rawdata.py

import logging
from pathlib import Path
import pandas as pd
import glob


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    for file_name in glob.glob(input_filepath + "/*.csv"):
        df_raw_in = pd.read_csv(file_name)
    
    
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir     = Path().resolve().parents[0]
    input_filepath  = str(project_dir/'data/raw')
    output_filepath = str(project_dir/'loan_prediction_repo/data/processed')

    main(input_filepath,output_filepath)



df_raw_in1 = pd.read_csv('C:/Users/jpatr_000/loan_prediction_repo/data/raw/trainingData.csv')
