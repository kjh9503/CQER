# KLQR

### Introduction

Knowledge-aware Logical Query-based Recommendation (KLQR) is a knowledge-aware recommendation framework, which consists of four components: (1) Query Generation Module, (2) Embedding Module (3) Logical Reasoning Module, (4) Prediction Module.

### Files

- `data/`
  - `MovieLens1M/`    
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG
    - `kg.txt`: knowledge graph file
	- `ratings.dat`: raw rating file of MovieLens1M
  - `BookCrossing/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG
    - `kg.txt`: knowledge graph file
	- `BX-Book-Ratings.csv`: raw rating file of BookCrossing
- `src/`
  - `data_preparation.py`: Proprocess raw data files
  - `dataloader.py`: dataloader for training and testing
  - `models.py`: colleciton of models (Embedding Module, Logical Reasoning Module, Prediction Module) in KLQR
  - `path_extration.py`: path extraction code for Query Generation Module
  - 'util.py': collection of helper funtions
	
- `main.py`: main code
  
### How to run the code

- MovieLens1M  
  ```
  $ python src/data_preparation.py -dataset MovieLens1M
  $ python src/path_extraction.py -dataset MovieLens1M
  $ CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test --valid_steps 100 --data_path data/MovieLens1M -n 128 -b 128 -d 32 -g 24 -lr 0.0001 --l2_lambda 1e-2 --geo beta --beta_mode "(1600,2)" --tasks "1p.2p.3p"
  ```

- BookCrossing 
  ```
  $ python src/data_preparation.py -dataset BookCrossing
  $ python src/path_extraction.py -dataset BookCrossing
  $ CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test --valid_steps 100 --data_path data/BookCrossing -n 128 -b 128 -d 32 -g 15 -lr 0.0001 --l2_lambda 1e-2 --geo beta --beta_mode "(800,2)" --tasks "1p.2p.3p"
  ```
