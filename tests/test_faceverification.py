from unittest import TestCase
from faceverification import recognition as rc
import torch
import numpy as np
from PIL import Image
import os


class TestRecognition(TestCase):
    def setUp(self):
        self.model = rc.get_model()

    def test_embedding(self):
        """Testing the embedding of an example image."""
        img = rc.load_image("images/1.PNG")
        result = rc.get_embedding(img, rc.get_model())
        actual = torch.load("tests/embedding_umur.pt")
        self.assertTrue((result == actual).sum())

