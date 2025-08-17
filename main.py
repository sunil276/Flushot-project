import pipelines.test_data_data_processing as test_data_data_processing
import pipelines.data_processing_pipeline_train as data_processing_pipeline_train
import pipelines.model_pipeline as model_pipeline


def main():
    train_data=data_processing_pipeline_train.run()
    test_data=test_data_data_processing.run()
    model_pipeline.run()

    


if __name__ == "__main__":
    main()