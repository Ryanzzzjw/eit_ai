from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.raw_data.raw_samples import load_samples
from eit_ai.train_utils.metadata import MetaData
import matplotlib.pyplot as plt
import logging
import seaborn as sns

logger = logging.getLogger(__name__)


def eval_dataset() -> None:
    metadata = MetaData()
    metadata.set_ouput_dir(training_name='Std pytorch test', append_date_time=True)
    
    metadata.set_4_raw_samples(data_sel=['Xih', 'Yih'])
    # metadata.set_4_raw_samples(data_sel=['Xih-Xh', 'Yih-Yh'])
    # metadata.set_4_raw_samples(data_sel=['Xih-Xh/Xh', 'Yih-Yh/Yh'])
    
    raw_samples = load_samples(MatlabSamples(), path, metadata)
    x = raw_samples.X.flatten()
    y = raw_samples.Y.flatten()

    # plt.violinplot(x)
    # # plt.show()
    # plt.violinplot(y)
    # # plt.show()
    fig, axs = plt.subplots(2)
    axs[0].violinplot(x)
    axs[0].set_title('X')
    axs[1].violinplot(y)
    axs[1].set_title('Y')
    
    


if __name__ == "__main__":
    from glob_utils.log.log import change_level_logging, main_log
    import logging
    main_log()
    change_level_logging(logging.DEBUG)

    debug = True

    if debug:
        path = r'C:\Users\ryanzzzjw\Downloads/eit_ai/datasets/20210929_082223_2D_16e_adad_cell3_SNR20dB_50k_dataset' \
               '/2D_16e_adad_cell3_SNR20dB_50k_infos2py.mat '
    else:
        path = ''

    eval_dataset()
    plt.show()

