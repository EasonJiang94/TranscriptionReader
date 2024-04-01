import pandas as pd
from loguru import logger
import re
import os
import subprocess
from dataloader import DataLoader
# pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)



#     def extract_by_llama(self):
#         prompt = "Analyze the information step by step, and tell me the person's age by the decription of a patient I gave. Just tell me the age, do not explain:\n"
#         prompt = "Analyze the information step by step, Do a simple summarize, then tell me the patient's age. Please output the age information only, do not tell me other information\n"
#         post_prompt = "\nAgain, Please output the digit of age information only, do not tell me other information. If you don't know the age information, return 'NaN', do not guess."
#         post_prompt = "\nAnswer with the format : 'Age : <num>'.\n If you don't know the answer, you can say : 'Age : Unkown'"
#         for cnt, trans in enumerate(self.data["transcription"]):
#             trans = trans.replace('"', '\\"')
#             token = prompt + trans + post_prompt
#             # print(token)
#             # os.system(f"llm -m l2c \"{token}\"")
#             response = subprocess.getoutput(f"llm -m l2c \"{token}\"")
#             print(f"{cnt}\t|\t{response}", flush=True)
#             if cnt <=18:
#                 continue
class FeatureExtractorInterface:
    AGE = "extracted_age"
    TREAT = "extracted_treatment"
    def __init__(self, dataloader:DataLoader):
        self.dl = dataloader
        self.data = dataloader.data

    def do_extract(self):
        self.data[FeatureExtractorInterface.AGE] = self.data[DataLoader.DES_STR].apply(self.extract_age)\
                                    .fillna(self.data[DataLoader.TRAN_STR].apply(self.extract_age))
        
        self.data[FeatureExtractorInterface.TREAT] = self.data[DataLoader.TRAN_STR].apply(self.extract_treatment)
        return self.data
    
    def extract_age(self, text):
        return None
    
    def extract_treatment(self, text):
        return None
    
    def show(self, n=100):
        logger.info(self.data[FeatureExtractorInterface.AGE].head(n))


class BasicExtractor(FeatureExtractorInterface):
    def __init__(self, dataloader:DataLoader, method='basic'):
        super().__init__(dataloader)
        # self.dl.show(4998, col=DataLoader.TRAN_STR)

    def do_extract(self):
        self.data[FeatureExtractorInterface.AGE] = self.data[DataLoader.DES_STR].apply(self.extract_age)\
                                    .fillna(self.data[DataLoader.TRAN_STR].apply(self.extract_age))\
                                    .fillna(self.data[DataLoader.TRAN_STR].apply(self.extract_y_o_age))\
                                    .fillna(self.data[DataLoader.TRAN_STR].apply(self.extract_special_age))
        
        self.data[FeatureExtractorInterface.TREAT] = self.data[DataLoader.TRAN_STR].apply(self.extract_treatment)
        return self.data

    def extract_age(self, text):
        # Ensure the input is a string; if not, return None
        if not isinstance(text, str):
            return None
        
        # For cases of ["56-year-old", "56-year old", "67-year old", "64 year-old", "2-1/2-year-old", "5-1/2 years old"]
        pattern = r'(\d+)(?:-(\d+)/(\d+))?\s*-?\s*year\s*s?\s*\s*-?\s*old'
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        
        if matches:
            # Extract whole number part and optional fraction part
            whole_part, numerator, denominator = matches[0]
            age = float(whole_part)
            if numerator and denominator:
                fraction = float(numerator) / float(denominator)
                age += fraction
            return age
        else:
            return None
    
    def extract_special_age(self, text):
        if not isinstance(text, str):
            return None
        # For cases ["46 years old", "ten year old", "one year old"]
        pattern = r'\b(?:almost\s)?(?:\d{1,2}(?:-1/2)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s(?:year|years)[\s-](?:and\s(?:a\s)?half\s)?old\b'
        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
            'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
        }

        # Convert the matched phrases to numeric age
        num = []
        for match in matches:
            parts = match.split()
            for part in parts:
                if part.isdigit():
                    num.append(int(part))
                elif part.lower() in word_to_num:
                    num.append(word_to_num[part.lower()])
                

        return num[0] if num else None
    
    def extract_y_o_age(self, text):
        # Direct regex to match "y/o" age format specifically
        if not isinstance(text, str):
            return None
        y_o_pattern = r'(\d+)\s*y/o'
        match = re.search(y_o_pattern, text)
        if match:
            return int(match.group(1))
        else:
            return None
    
class Llama2Extractor(BasicExtractor):
    def __init__(self, dataloader:DataLoader, method='basic'):
        super().__init__(dataloader)
        # self.dl.show(4998, col=DataLoader.TRAN_STR)

    def do_extract(self):
        self.data[FeatureExtractorInterface.AGE] = self.data[DataLoader.TRAN_STR].apply(self.extract_age)        
        self.data[FeatureExtractorInterface.TREAT] = self.data[DataLoader.TRAN_STR].apply(self.extract_treatment)
        return self.data

    def extract_age(self, text):
        # prompt = "Analyze the information step by step, and tell me the person's age by the decription of a patient I gave. Just tell me the age, do not explain:\n"
        prompt = "Analyze the information step by step, Do a simple summarize, then tell me the patient's age. Please output the age information only, do not tell me other information\n"
        post_prompt = "\nAgain, Please output the digit of age information only, do not tell me other information. If you don't know the age information, return 'NaN', do not guess."
        post_prompt = "\nAnswer with the json format : {'Age' : <num or None>}.\n If you don't know the answer, you can say : 'Age : Unkown'"
        post_prompt = ""
        text = text.replace('"', '\\"')
        token = prompt + text + post_prompt
        # print(token)
        # os.system(f"llm -m l2c \"{token}\"")
        response = subprocess.getoutput(f"llm -m l2c \"{token}\"")
        print(f"{response}", flush=True)
        return response
   

class Extractor(FeatureExtractorInterface):
    
    def __init__(self, dataloader:DataLoader, method='basic'):
        super().__init__(dataloader)
        # self.dl.show(4998, col=DataLoader.TRAN_STR)
        self.extract_methods = {
            "basic" : BasicExtractor(dataloader),
            "llama2" : Llama2Extractor(dataloader)
        }
        self.extractor = self.extract_methods[method]
        self.extractor.do_extract()
        # self.show()



if __name__ == "__main__":
    path = "../../data/mtsamples.csv"
    dl = DataLoader(path)
    ex = Extractor(dl, method="llama2")
    test_cases = ["56-year-old", "56-year old", "67-year old", "64 year-old", "2-1/2-year-old", "5-1/2 years old"]
    extracted_ages = [BasicExtractor(dl).extract_age(case) for case in test_cases]
    print(extracted_ages)
    test_cases = ["46 years old", "ten year old", "one year old"]
    extracted_ages = [BasicExtractor(dl).extract_special_age(case) for case in test_cases]
    print(extracted_ages)
    test_cases = ["This 31 y/o"]
    extracted_ages = [BasicExtractor(dl).extract_y_o_age(case) for case in test_cases]
    print(extracted_ages)
    ex.show(21)
    # print(dl.columns)