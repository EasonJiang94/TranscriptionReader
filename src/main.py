from feature_extractor.dataloader import DataLoader
from feature_extractor.extractors import Extractor
import argparse
from loguru import logger
def main():
    data_path = "../data/mtsamples.csv"
    parser = argparse.ArgumentParser(description='Model Selector Script')
    parser.add_argument('-m', '--model', choices=['basic', 'llama2', 'ChatGPT'], required=True, help='Specify the model to use. Options: basic, llama2, ChatGPT.')
    parser.add_argument('-s', '--size', type=int, default=20, help='Specify the size of analyzed data to display. Default is 20.')

    args = parser.parse_args()
    logger.info(f'You have chosen the "{args.model}" model.')

    dataloader = DataLoader(data_path)
    ex = Extractor(dataloader, method=args.model, size=args.size)
    ex.show(n=20, features = [Extractor.AGE, Extractor.TREAT])


if __name__ == "__main__":
    main()