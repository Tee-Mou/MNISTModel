import os
from DataOps import DataManager

def main():
    # manager = DataManager(model="EuroSAT", batch_size=100, name="EuroSAT95")
    # manager.train(csv_name="EuroSAT Retrain", lr=0.001)
    DataManager.plot_training_results(csv_name="EuroSAT Retrain")
    return 0
    
if __name__ == "__main__":
    main()