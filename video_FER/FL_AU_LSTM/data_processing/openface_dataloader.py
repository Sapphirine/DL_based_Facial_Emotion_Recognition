
import os
from .csv_dataloader import open_and_extract

def _extract_columns(row, wanted_columns):
    
    return {col : float(row[col]) for col in wanted_columns if col in row}

def _extract_AU_activations(row):
    
    return {row['timestamp'] : _extract_columns(row, 
                                ['AU' + "{0:02d}".format(k) + '_c' for k in range(46)])}

def _extract_AUs(row):
    
    return {row['timestamp'] : _extract_columns(row, 
                                ['AU' + "{0:02d}".format(k) + '_r' for k in range(46)])}

def _extract_2Dlandmarks(row):
   
    return {row['timestamp'] : _extract_columns(row, ['x_' + str(k) for k in range(68)]
                                + ['y_' + str(k) for k in range(68)])}

def get_2Dlandmarks(filename):
   
    return open_and_extract(filename, _extract_2Dlandmarks)

def get_AUs(filename):
   
    return open_and_extract(filename, _extract_AUs)

def get_AU_activations(filename):
 
    return open_and_extract(filename, _extract_AU_activations)

def load_OpenFace_features(root_dirname, features='AUs'):
    
    output={}

    for dirname, _, file_list in os.walk(root_dirname):
        for filename in file_list:
            if filename.endswith(".txt"):
                record_id = filename[0:9]
                filename = os.path.join(dirname,filename)

                output.update({record_id : globals()['get_' + features](filename)})
    
    return output
