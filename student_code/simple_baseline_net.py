import torch.nn as nn
from external.googlenet.googlenet import googlenet
import ipdb
import torch

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, num_q_words, num_a_words): # 2.2 TODO: add arguments needed
        super().__init__()
	    ############ 2.2 TODO

        self.num_q_words = num_q_words
        self.num_a_words = num_a_words

        # self.googlenet_features = googlenet() #pretrained
        # # N x 1024 x 7 x 7
        # self.image_feature_extractor = nn.Linear(1024*49, self.num_q_words)
        self.image_feature_extractor = googlenet(pretrained=True)

        self.word_feature_extractor = nn.Linear(self.num_q_words, 1024)

        self.classifier = nn.Linear(1024 * 2, self.num_a_words)
	    ############

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO

        # ipdb.set_trace()
        # image_features = self.googlenet_features(image)
        # image_features = self.image_feature_extractor(image_features.view(image_features.size(0), -1))
        image_features = self.image_feature_extractor(image).squeeze()

        word_features = self.word_feature_extractor(question_encoding)
        
        return self.classifier(torch.cat([image_features , word_features], dim=1))

	    ############
        # raise NotImplementedError()
