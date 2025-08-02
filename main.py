import test_data_data_processing
import data_processing_pipeline_train


def main():
    train_data=data_processing_pipeline_train.run()
    test_data=test_data_data_processing.run()
    


if __name__ == "__main__":
    main()