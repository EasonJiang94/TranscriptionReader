import pandas as pd
from loguru import logger
import re
import os
import subprocess
import json
from dataloader import DataLoader
# pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

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
    
    def show(self, feature=None, n=100):
        if feature is None:
            feature = FeatureExtractorInterface.AGE
        logger.info(self.data[feature].head(n))


class BasicExtractor(FeatureExtractorInterface):
    def __init__(self, dataloader:DataLoader, method='basic'):
        super().__init__(dataloader)
        # self.dl.show(4998, col=DataLoader.TRAN_STR)
        logger.info("Launched BasicExtractor")
        

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
    # follow the instruction from the link below to use llm CLI
    # https://simonwillison.net/2023/Aug/1/llama-2-mac/
    def __init__(self, dataloader:DataLoader, size=5, method='basic'):
        super().__init__(dataloader)
        # self.dl.show(4998, col=DataLoader.TRAN_STR)
        logger.info("Launched Llama2Extractor")
        self.cnt = 0
        self.size = size

    def do_extract(self):
        #self.data[FeatureExtractorInterface.AGE] = self.data[DataLoader.TRAN_STR].apply(self.extract_age)        
        self.data[FeatureExtractorInterface.TREAT] = self.data[DataLoader.TRAN_STR].apply(self.extract_treatment)
        return self.data

    def extract_age(self, text):
        if not isinstance(text, str):
            return None
        if self.cnt > self.size:
            return None
        # prompt = "Analyze the information step by step, and tell me the person's age by the decription of a patient I gave. Just tell me the age, do not explain:\n"
        prompt = "Analyze the information step by step, Do a simple summarize, then tell me the patient's age. Please output the age information only, do not tell me other information\n"
        # post_prompt = "\nAgain, Please output the digit of age information only, do not tell me other information. If you don't know the age information, return 'NaN', do not guess."
        post_prompt = "\nAnswer with the json format : {'Age' : <num or None>}.\n If you don't know the answer, you can say : 'Age : Unkown'"
        # post_prompt = ""
        text = text.replace('"', '\\"')
        token = prompt + text + post_prompt
        # print(token)
        # os.system(f"llm -m l2c \"{token}\"")
        response = subprocess.getoutput(f"llm -m l2c \"{token}\"")
        print(f"{self.cnt}\t| {response}", flush=True)
        self.cnt += 1
        return response
    
    def extract_treatment(self, text):
        if not isinstance(text, str):
            return None
        if self.cnt > self.size:
            return None
        logger.debug("Extracting treatment information")
        prompt = "Analyze the information step by step, make a brief summarization of what treatment of the patient received\n"
        post_prompt = "Answer it in 150 words"
        # post_prompt = ""
        text = text.replace('"', '\\"')
        token = prompt + text + post_prompt
        # print(token)
        # os.system(f"llm -m l2c \"{token}\"")
        response = subprocess.getoutput(f"llm -m l2c \"{token}\"")
        print(f"{self.cnt}\t| {response}", flush=True)
        self.cnt += 1
        return response
    
class ChatGPTExtractor(BasicExtractor):
    # follow the instruction from the link below to use chat-gpt api
    # https://platform.openai.com/docs/quickstart?context=python
    
    def __init__(self, dataloader:DataLoader,size=5, method='basic'):
        from openai import OpenAI
        super().__init__(dataloader)
        # self.dl.show(4998, col=DataLoader.TRAN_STR)
        self.client = OpenAI()
        logger.info("Launched Llama2Extractor")
        self.cnt = 0
        self.size = size
        


    def do_extract(self):
        # self.data[FeatureExtractorInterface.AGE] = self.data[DataLoader.TRAN_STR].apply(self.extract_age)        
        self.data[FeatureExtractorInterface.TREAT] = self.data[DataLoader.TRAN_STR].apply(self.extract_treatment)
        return self.data

    def extract_age(self, text):
        if not isinstance(text, str):
            return None
        if self.cnt > self.size:
            return None
        prompt = "Analyze the information step by step, Do a simple summarize, \
            then tell me the patient's age. \
            Please output the age information only, do not tell me other information\n\
            Answer in a json format, the key of age is \"age\"\n"
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        ans = response.choices[0].message.content
        try:
            ans.replace("\\n", "")
            ans = json.loads(ans)["age"]
        except:
            logger.warning(f"it could not be analyze by json lib\n{ans = }")
            pass
        print(f"{self.cnt}\t| {ans}", flush=True)
        self.cnt += 1

        return ans
    
    def extract_treatment(self, text):
        if not isinstance(text, str):
            return None
        if self.cnt > self.size:
            return None
        prompt = "Analyze the information step by step, \
                make a brief summarization of what treatment of the patient received\n"
        
        post_prompt = "\nPlease answer it in 100 words."
        token = text + post_prompt
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": token}
            ]
        )
        ans = response.choices[0].message.content
        
        print(f"{self.cnt}\t| {ans}", flush=True)
        self.cnt += 1

        return ans
   

class Extractor(FeatureExtractorInterface):
    EXTRACTOR_LIST = ["basic", "llama2", "ChatGPT"]
    def __init__(self, dataloader:DataLoader, method='basic'):
        super().__init__(dataloader)
        # self.dl.show(4998, col=DataLoader.TRAN_STR)
        if method not in Extractor.EXTRACTOR_LIST:
            logger.error(f"{method = } is not supported")
            raise ValueError
        self.extract_methods = {
            "basic" : BasicExtractor,
            "llama2" : Llama2Extractor,
            "ChatGPT" : ChatGPTExtractor
        }
        self.extractor = self.extract_methods[method](dataloader)
        self.extractor.do_extract()
        # self.show()



if __name__ == "__main__":
    path = "../../data/mtsamples.csv"
    dl = DataLoader(path)
    # EXTRACTOR_LIST = ["basic", "llama2", "ChatGPT"]
    ex = Extractor(dl, method="ChatGPT")
    # test_cases = ["56-year-old", "56-year old", "67-year old", "64 year-old", "2-1/2-year-old", "5-1/2 years old"]
    # extracted_ages = [BasicExtractor(dl).extract_age(case) for case in test_cases]
    # print(extracted_ages)
    # test_cases = ["46 years old", "ten year old", "one year old"]
    # extracted_ages = [BasicExtractor(dl).extract_special_age(case) for case in test_cases]
    # print(extracted_ages)
    # test_cases = ["This 31 y/o"]
    # extracted_ages = [BasicExtractor(dl).extract_y_o_age(case) for case in test_cases]
    # print(extracted_ages)
    ex.show(feature=FeatureExtractorInterface.TREAT, n=21)
    # print(dl.columns)
