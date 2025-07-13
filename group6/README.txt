Group 6 README

STEP 1: Setting up environments

    # The first step is to create environments to ensure package compatibility

    install conda

    create bodmas_env

        Download the data from https://whyisyoung.github.io/BODMAS/
        Place bluehex.npz into the "~/group6/BODMAS/code/multiple_data" directory

        conda create -n bodmas_env python=3.6.8
        conda activate bodmas_env
        cd BODMAS/code
        pip install -r requirements.txt
        python setup.py install
        conda deactivate
        cd ../..

    create tddpm environment
        export REPO_DIR=/path/to/group6/tab-clone
        cd $REPO_DIR

        conda create -n tddpm python=3.9.7
        conda activate tddpm

        pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r requirements.txt

        # if the following commands do not succeed, update conda
        conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
        conda env config vars set PROJECT_DIR=${REPO_DIR}

        conda deactivate
        cd ..

    create sklearn_env
        conda create -n sklearn_env
        conda activate sklearn_env
        pip install --upgrade pip
        pip install scikit-learn
        conda deactivate

STEP 2: Extracting the BODMAS dataset

    # Then you need to load the BODMAS data into the project space and extract the data

    cd BODMAS/code
    conda activate bodmas_env
    ./main_bluehex_multiclass.sh
    # This creates a log in "~/group6/BODMAS/code/logs/bluehex_multiclass", check there to wait for finish
    Change line 114 in "~/group6/BODMAS/code/bluehex_main.py" from save=True to save=False
    conda deactivate

STEP 3: Synthetic Generation with Tab-DDPM 

    # Next we utilize formatter.py to provide splits for Tab-DDPM, then generate the synthetic data

    cd ../../tab-clone
    conda activate tddpm
    python data/bodmas/formatter.py

    python scripts/swap_helper.py s
    python scripts/pipeline.py --config exp/bodmas/ddpm_sf_tune_best/config.toml --train --sample --eval

    python scripts/swap_helper.py m
    python scripts/pipeline.py --config exp/bodmas/ddpm_mf_tune_best/config.toml --train --sample --eval

    python scripts/swap_helper.py l
    python scripts/pipeline.py --config exp/bodmas/ddpm_lf_tune_best/config.toml --train --sample --eval

    python scripts/ds_merge.py 1
    conda deactivate

STEP 4: Running BODMAS on the augmented dataset

    cd ../BODMAS/code
    conda activate bodmas_env
    ./main_bluehex_multiclass.sh
    # Wait for the script to finish
    conda deactivate

STEP 5: Calculate metrics

    conda activate sklearn_env

    python pred_helper.py base 
    python pred_helper.py augmented

    conda deactivate

    # The data has been output to the console or you can check 
    # "~/BODMAS/code/multiple_data/g6data/predictions/pred_{identifier}/"
    # To find the output macro f1 score

OTHER NOTES:

    Command to tune tabDDPM:
        python scripts/tune_ddpm.py bodmas {# samples in training split} synthetic mlp ddpm_tune
        eval_mlp.py has a hardcoded path at line 83

        Command for sized tunings:
            python scripts/tune_ddpm.py bodmas #### synthetic mlp ddpm_sf_tune
            python scripts/tune_ddpm.py bodmas #### synthetic mlp ddpm_mf_tune
            python scripts/tune_ddpm.py bodmas #### synthetic mlp ddpm_lf_tune

    If you accidentally run bodmas
        >ps
        look for the PID of python under sh
        >kill $PID

    Information about auxiliary scripts can be found in AssortedHelpers/helpers_readme.txt 