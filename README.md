# Cat detector

## Introduction

Ths is my attempt to reproduce results from the Viola Jones 
paper applied to cat detection. The idea was to estimate the gap
between the paper and a new working implementation on another use cas



## Details

1. Download data to workspace (CAT_WORKSPACE)
2. Create dataframe from downloaded folders
3. Visualisation images from the dataframe
4. Training
    1. Extract positive patches from images
    2. For each stage
        1. Extract negative patches (GT and for cascade) from images
        2. Learn Haar model via random search of haar feature
        3. Update cascade zith new model
5. Testing
    1. Generate image visualisation with detected cat and other detection
    
    
# TODO

- [ ] Cascade utils: predict and filter (use sklearn Pipeline?)
- [ ] CLI
- [ ] Download original data script
- [ ] SURF and LBP feature integration
- [ ] Use other classifier during clf search
- [ ] Python packaging
- [ ] Add n_scales parameters and steps for each one
- [ ] create config management ( dotenv and ini file)
- [ ] add debug tools


Logging + Log visu + Log perf --> check mlflow?

