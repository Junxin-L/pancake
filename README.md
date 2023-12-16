# Pancake in Python
## Overview

This Python implementation replicates and extends the concepts from the paper "Pancake: Frequency Smoothing for Encrypted Data Stores". It **enhances the original Pancake model by adding a correlation between queries**. This feature allows for more sophisticated analysis and understanding of query patterns over encrypted data stores.

## Features

- **Pancake Model Replication**: Implements the core ideas of frequency smoothing in encrypted data stores as described in the original Pancake paper.
- **Query Correlation**: Extends the model to include correlation between queries, enabling deeper insights into data access patterns.
- **Frequency Calculation Enhancements**: Offers improved methods for calculating frequencies and correlations within the data store.

## Requirements

- Python 3.x
- NumPy

## Installation

Simply copy the `pancake.py` file to your project directory. Ensure that you have Python and NumPy installed.

## Usage

To use the Pancake class, import it into your Python script:

`from pancake import Pancake`

Create an instance of the Pancake class:

`pancake_instance = Pancake(num_states, s=2, smoothing=0.0, seed=None, add_rep=True, delta=-1)`

- `num_states`: Number of states (or keys) in the data store.
- `s`: Stage (default is 2).
- `smoothing`: Smoothing parameter (default is 0.0).
- `seed`: Random seed (optional).
- `add_rep`: Boolean to decide whether to add replicas to keys (default is True).
- `delta`: Delta parameter for correlation (default is -1).

The instance provides various methods for frequency calculations, adding replicas, and generating query sequences.

## Methods

- `cal_corr()`: Calculates the correlation between queries.
- `gen_init()`: Generates initial probabilities for keys.
- `gen_transmat()`: Creates a random transition matrix.
- `cal_freq()`: Computes frequency of each key.
- `add_rep()`: Adds replicas to keys to modify their frequencies.
- `correlated_query_seq()`: Generates a sequence of queries with correlation.
