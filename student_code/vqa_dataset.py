from decimal import InvalidContext
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from external.vqa.vqa import VQA

import numpy as np
import scipy.stats
import torchvision
import ipdb
from tqdm import tqdm

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        # Create the question map if necessary
        self.questions_dict = {}
        sentences = [] 
        for d in self._vqa.questions["questions"]:
            sentences.append(d["question"])

        if question_word_to_id_map is None:
            ############ 1.6 TODO

            # sentences = [d["question"] for d in self._vqa.questions["questions"]]
            wordlist = self._create_word_list(sentences)
            self.question_word_to_id_map = self._create_id_map(wordlist, self.question_word_list_length)

            ############
            # raise NotImplementedError()
        else:
            self.question_word_to_id_map = question_word_to_id_map

        for d in tqdm(self._vqa.questions["questions"]):            
            question_str = d["question"]
            # question_result = torch.zeros(20, self.question_word_list_length)
            # for i, word in enumerate(question_str.split(" ")):
            #     wordlower = word.lower()
            #     if wordlower in self.question_word_to_id_map.keys():
            #         question_result[i][self.question_word_to_id_map[wordlower]] = 1.
            self.questions_dict[d["question_id"]] = question_str # question_result # d["question"]

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO

            # could be correct to either concat answers together, or choose more popular answer .. the solution apparently uses majority

            answers = []
            for qid in self._vqa.getQuesIds():
                subresult = [ans["answer"] for ans in self._vqa.loadQA(qid)[0]["answers"]]
                answers.append(" ".join(subresult))

            wordlist = self._create_word_list(answers)
            self.answer_to_id_map = self._create_id_map(wordlist, answer_list_length)

            ############
            # raise NotImplementedError()
        else:
            self.answer_to_id_map = answer_to_id_map


    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """

        ############ 1.4 TODO
        result = []
        for sentence in sentences:
            tmp = sentence.lower()

            # remove invalid chars
            i = 0
            while i < len(tmp):
                while i < len(tmp) and (not (tmp[i].isalpha() or tmp[i] == " ")) :
                    tmp = tmp[:i] + tmp[i+1:]
                i += 1
            # tmp = [s for s in tmp if (s.isalpha() or s == " ")]

            tmp = str(tmp)
            tmp = tmp.split(" ")
            result+= tmp
        return result


        ############
        # raise NotImplementedError()


    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """

        ############ 1.5 TODO

        # stackoverflow.com/questions/48784908/numpy-unique-sort-based-on-counts
        unique_strings, string_counts = np.unique(word_list, return_counts=True)
        index_sort = np.argsort(-string_counts)
        index_sort = index_sort[:max_list_length]
        unique_strings = unique_strings[index_sort]
        return {str_ : i for i, str_ in enumerate(unique_strings)}

        ############
        # raise NotImplementedError()


    def __len__(self):
        ############ 1.8 TODO
        return len(self._vqa.getQuesIds())


        ############
        # raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        ques_id = self._vqa.getQuesIds()[idx]

        ############

        if self._cache_location is not None and self._pre_encoder is not None:
            image_id = self._vqa.loadQA(ques_id)[0]["image_id"]
            # 12 digits of numeric
            zeros = "0" * (12 - len(str(image_id)))
            image_id_str = zeros + str(image_id)

            # the caching and loading logic here
            feat_path = os.path.join(self._cache_location, f'{image_id_str}.pt')
            try:
                image = torch.load(feat_path)
            except:
                image_path = os.path.join(
                    self._image_dir, self._image_filename_pattern.format(image_id_str))
                image = Image.open(image_path).convert('RGB')
                image = self._transform(image).unsqueeze(0)
                image = self._pre_encoder(image)[0]
                torch.save(image, feat_path)
        else:
            ############ 1.9 TODO
            # load the image from disk, apply self._transform (if not None)

            # todo use imgstrpattern, imgdir, etc (don't hardcode)
            image_id = self._vqa.loadQA(ques_id)[0]["image_id"]

            # 12 digits of numeric
            zeros = "0" * (12 - len(str(image_id)))
            # image_filename = f"{self._image_dir}/COCO_train2014_{zeros}{image_id}.jpg"
            # image_id_str = zeros + str(idx//10)
            image_id_str = zeros + str(image_id)

            image_filename = os.path.join(
                self._image_dir, self._image_filename_pattern.format(image_id_str))
                # self._image_dir, self._image_filename_pattern.format(idx//10))

            image = Image.open(image_filename).convert('RGB')
            # ipdb.set_trace()
            # print(image.size, "imagesize")
            # print(np.asarray(image).shape, "np asarray shape")
            # if image.size[0] != 3:
            #     print("invalid!!", image.size)
                # ipdb.set_trace()
            if self._transform:
                image = self._transform(image)
            else:
                assert False
                image = torchvision.transforms.ToTensor()(image)

            ############
            # raise NotImplementedError()

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors
        answers = scipy.stats.mode([ans["answer"] for ans in self._vqa.loadQA(ques_id)[0]["answers"]])
        answers = answers[0].item()
        # print("answers", answers)
        # print("idmap result", self.answer_to_id_map[answers])
        answers_tensor = torch.zeros(self.answer_list_length)
        if answers in self.answer_to_id_map.keys():
            answers_tensor[self.answer_to_id_map[answers]] = 1.

        question_str = self.questions_dict[ques_id]
        question_result = torch.zeros(20, self.question_word_list_length)
        for i, word in enumerate(question_str.split(" ")[:20]):
            wordlower = word.lower()
            if wordlower in self.question_word_to_id_map.keys():
                question_result[i][self.question_word_to_id_map[wordlower]] = 1.
        question_tensor = question_result

        # question_tensor = torch.zeros(self.question_word_list_length)
        # for word in questions:
        #     i = self.question_word_to_id_map[q]
        #     question_tensor[i] = 1.

        ############
        result =  {
            'idx': idx,
            'image': image, #.to(device = ("cuda" if torch.cuda.is_available() else "cpu")),
            'question': question_tensor,#.to(device = ("cuda" if torch.cuda.is_available() else "cpu")),
            'answers': answers_tensor,#.to(device = ("cuda" if torch.cuda.is_available() else "cpu"))
        }
        # print(result)
        # ipdb.set_trace()
        # if image.shape[0] != 3:
        #     return None
        return result
