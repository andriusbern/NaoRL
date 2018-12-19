"""
Script for running trained models

The custom nao_rl trained models can be run either on virtual NAO in V-REP or 
on the real Robot

Positional arguments:
    1. Tensorflow weights file
    2. []
"""

from argparse import ArgumentParser
import tensorflow as tf
import nao_rl
import gym

