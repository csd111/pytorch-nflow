import os
import torch
from datetime import datetime
from models import RealNVP, Glow
from training import Trainer


if __name__ == '__main__':

    # Path to the database
    data_dir = "/path/to/training/database/"

    # The device (GPU/CPU) on which to execute the code
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # The model to train
    model = Glow(context_blocks=4, flow_steps=8, input_channels=3,
                 hidden_channels=256, quantization=256, lu_decomposition=False)

    model.to(device)

    # Path to the directory where the results will be saved
    saving_directory = \
        os.path.join(os.getcwd(), 'results',
                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    trainer = Trainer(model=model,
                      data_path=data_dir,
                      batch_size=4,
                      learning_rate=0.0001,
                      weight_norm=True,
                      saving_directory=saving_directory,
                      device=device)         

    # Start the learning
    trainer.run()
