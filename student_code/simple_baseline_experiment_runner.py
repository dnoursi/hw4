from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torchvision
import torch
import ipdb

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation):

        ############ 2.3 TODO: set up transform

        # want N x 3 x 224 x 224

        transform = torchvision.transforms.Compose([
            # todo: normalize to 0,1?
            # todo: tranpose axes?
            # torchvision.transforms.Resize((3,224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        ############

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   transform=transform,
                                   ############ 2.4 TODO: fill in the arguments
                                   question_word_to_id_map=None,
                                   answer_to_id_map=None,
                                   ############
                                   )
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 transform=transform,
                                 ############ 2.4 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map,
                                 answer_to_id_map=train_dataset.answer_to_id_map,
                                 ############
                                 )

        # model = SimpleBaselineNet(num_words=len(train_dataset.question_word_to_id_map))
        model = SimpleBaselineNet(
            num_q_words=train_dataset.question_word_list_length,
            num_a_words=train_dataset.answer_list_length,
        )

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

        ############ 2.5 TODO: set up optimizer

        word_parameters = list(model.word_feature_extractor.parameters()) + \
        list(model.image_feature_extractor.parameters())

        # word_parameters = list(model.word_feature_extractor.parameters()) 

          #list(model.googlenet_features.parameters()) + \

        self.word_optimizer = torch.optim.SGD(word_parameters, lr=0.8, momentum=0.9)
        self.softmax_optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)


        ############


    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 2.7 TODO: compute the loss, run back propagation, take optimization step.

        # loss = predicted_answers - true_answer_ids
        # loss = loss * loss
        # loss = loss.norm()
        loss = torch.nn.CrossEntropyLoss()(predicted_answers, true_answer_ids)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=20)
        # self._model.weight.data = torch.clamp(self._model.weight.data , -1500, 1500)



        self.word_optimizer.zero_grad()
        self.softmax_optimizer.zero_grad()
        loss.backward()

        # clip gradnorm now!!

        self.word_optimizer.step()
        self.softmax_optimizer.step()
        return loss

        ############ 
        # raise NotImplementedError()
