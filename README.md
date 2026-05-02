# anphy-sleep-stage-classification

This is the repository for our CS 7643 project on
classifying sleep stages using the ANPHY-Sleep dataset.
  
This is the file structure of this repository:

```
└── 📁anphy-sleep-stage-classification
    └── 📁anphy_sleep_data
        └── 📁patient_records
        ├── Data Preprocessing.ipynb
        ├── download_patient_data.py
    └── 📁checkpoints
    └── 📁configs
        ├── CNN_ADAM.yaml
        ├── CNN_base.yaml
        ├── LSTM_ADAM_only_body.yaml
        ├── LSTM_ADAM.yaml
        ├── LSTM_base.yaml
        ├── LSTM_context_ADAM.yaml
        ├── LSTM_context.yaml
        ├── LSTM_only_body.yaml
    └── 📁models
        └── 📁__pycache__
        ├── __init__.py
        ├── CNN_only_body.py
        ├── CNN_with_context_only_body.py
        ├── CNN_with_context.py
        ├── LSTM_only_body.py
        ├── LSTM_with_context_only_body.py
        ├── LSTM_with_context.py
        ├── my_lstm_model.py
        ├── my_model.py
    └── 📁outfiles
        ├── CNN_ADAM-5152503.out
        ├── CNN_base-5152502.out
        ├── CNNWithBiLSTM_ADAM_only_body-5152352.out
        ├── CNNWithBiLSTM_ADAM_only_body-5157201.out
        ├── CNNWithBiLSTM_ADAM_only_body-5161698.out
        ├── CNNWithBiLSTM_ADAM-5148778.out
        ├── CNNWithBiLSTM_ADAM-5149990.out
        ├── CNNWithBiLSTM_ADAM-5150845.out
        ├── CNNWithBiLSTM_base-5149989.out
        ├── CNNWithBiLSTM_base-5150823.out
        ├── CNNWithBiLSTM_base-5152004.out
        ├── CNNWithBiLSTM_body_only-5149685.out
        ├── CNNWithBiLSTM_body_only-5150827.out
        ├── CNNWithBiLSTM_body_only-5151995.out
        ├── CNNWithBiLSTMContext_ADAM-5152411.out
        ├── CNNWithBiLSTMContext_ADAM-5157235.out
        ├── CNNWithBiLSTMContext_ADAM-5161770.out
        ├── CNNWithBiLSTMContext-5148779.out
        ├── CNNWithBiLSTMContext-5152771.out
        ├── CNNWithBiLSTMContext-5155621.out
        ├── CNNWithSingleDirectionalLSTM-5064652.out
        ├── original_CNN_before_midpoint.out
    └── 📁results_files
    └── 📁sbatch_files
        ├── CNN_ADAM.sbatch
        ├── CNN_base.sbatch
        ├── LSTM_ADAM_only_body.sbatch
        ├── LSTM_ADAM.sbatch
        ├── LSTM_base_only_body.sbatch
        ├── LSTM_base.sbatch
        ├── LSTMContext_ADAM.sbatch
        ├── LSTMContext.sbatch
    ├── .gitignore
    ├── EDA.ipynb
    ├── main.py
    ├── manual-env.txt
    ├── Model Testing.ipynb
    ├── pace_anphy_instructions.pdf
    ├── README.md
    ├── recording_epoch_nums_body_only.csv
    ├── recording_epoch_nums.csv
    └── requirements.txt
```

The results_file folders contain the predicted and actual results on test data for the model per seed.
They are formatted as follows:

```
└── 📁CNN_ADAM_optim_with_scheduler_seed_42
            ├── results.parquet
            ├── seed_42_test_loss.csv
            ├── seed_42_train_loss.csv
            ├── seed_42_val_loss.csv
```
Note: Not every model will have the loss.csvs, but most do. These are the losses per epoch.  
Note that if you actually want to use the sbatch files, you will need to move them back out
of their dedicated folder so that SLURM can figure out where the python files to run are.
I just moved them afterward for organization.