from forge.embeddings import Forge
from forge.processor import MIPProcessor
from forge.utils import check_true


def pretrain(forge: Forge,
             input_mip_folder, relaxation_list,
             output_mip_to_mipinfo_pkl, output_forge_pkl, output_log_file):

    _validate_args(forge)

    # MIP Processor
    mip_processor = MIPProcessor(seed=forge.seed)

    # Create MIP to MIPInfo dictionary
    mip_processor.convert_mip_to_mipinfo(input_mip_folder=input_mip_folder,
                                         output_mip_to_mipinfo_pkl=output_mip_to_mipinfo_pkl,
                                         relaxation_list=relaxation_list,
                                         has_return=False)

    # List of MIPInfo objects for training
    mipinfo_list = mip_processor.load_mipinfo_from_pickles([output_mip_to_mipinfo_pkl])

    # Pre-train the Forge model
    forge.pretrain(input_mipinfo_list=mipinfo_list,
                   output_forge_pkl=output_forge_pkl,
                   output_log_file=output_log_file)

def _validate_args(forge, check_trained=False):

    check_true(isinstance(forge, Forge), TypeError("Error: Forge input should be a Forge instance."))
    if check_trained:
        check_true(forge.is_trained, ValueError("Error: Forge has not been trained."))

    pass
    # # Train/test data
    # check_true(data is not None, ValueError("Data input cannot be none."))
    # check_true(isinstance(data, (str, pd.DataFrame)),
    #            TypeError("Data should be string of filepath or data frame."))
    #
    # if save_file is not None:
    #     check_true(isinstance(save_file, (bool, str)),
    #                TypeError("Save file should be boolean or a string filepath."))


