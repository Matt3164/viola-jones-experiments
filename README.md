# Cat detector

1. Download data to workspace (CAT_WORKSPACE)
2. Dataframe from folder
3. Visualisation from dataframe
4. Training
    1. Extract patches from images (positives and negatives)
    2. Learn Haar model (Select k best among haar features )
    3. Use model to select harder patches 
5. Testing
    1. Generate image visualistion with detected cat and other detection
    
    
# TODO

- [ ] Todo
- [ ] CLI
- [ ] Download original data script
- [ ] Iterator from DataFrame
- [ ] Preprocessing function : resize , cvtColor, remove mean and variance
- [ ] Path utils class to manage path to store images
- [ ] Asbtraction on used classifier and features
- [ ] SURF and LBP feature integration
- [ ] Use more than one features at a time for classifier

