##### For models evaluation #####

from draw_data import draw_MLsolver, draw_EIDORS_data, data_loading
import os

# if there is more measurement inputs than 1:
index = 0        # variable indicates which measurement to show starting from 0  

PATH = "C:\\Users\manta\Documents\Master thesis\Tr" #path to files 

#for 3054 elements (max mesh el. size od 0.05)
if 0:                         
    input_fNAME = 'Xih_3054x20.mat'
    output_fNAME = 'Yih_3054x20.mat'
    trianglesDATA_fNAME = 'tr_data_3054el.mat' # consist of both nodes coordinates and triangles nodes indexes
    model_NAME = "1st_700epochs_93p_acc.model" #file name to open or to save ML model

#for 990 elements  (max mesh el. size od 0.09)
if 1:
    input_fNAME = 'Xih_990x20.mat'
    output_fNAME = 'Yih_990x20.mat'
    trianglesDATA_fNAME = 'tr_data_990el.mat'  
    model_NAME = "model_autokeras_990el_allNORM"

input_PATH = os.sep.join([PATH, input_fNAME])
output_PATH = os.sep.join([PATH, output_fNAME])
trianglesDATA_fPATH = os.sep.join([PATH, trianglesDATA_fNAME])

# tensor board

inputDATA, outputDATA, trianglesDATA, model = data_loading(input_PATH, output_PATH, trianglesDATA_fPATH, model_NAME)

if 1:
    draw_EIDORS_data(outputDATA, trianglesDATA, meas_index = index)
if 1:
    draw_MLsolver(inputDATA, model, trianglesDATA,  meas_index = index)
