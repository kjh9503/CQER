# CQER

## Introduction

Conjunctive Query Embedding-based Few-shot Item Recommendation (CQER) is a knowledge-aware recommendation framework, which consists of four components: (1) Query Generation Module, (2) Embedding Module (3) Query Execution Module.

## Files

- `data/`
  - `BookCrossing/`: raw dataset of Book-Crossing
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG
    - `kg.txt`: knowledge graph file
    - `BX-Book-Ratings.csv`: raw rating file of BookCrossing
  - `MovieLens1M_0.1/`,`MovieLens1M_0.3/`,`MovieLens1M_0.5/`: preprocessed sparse dataset of MovieLens1M with respective sparsity ratio 10%, 30%, 50%
    - `train.txt`: train file
    - `valid.txt`: valid file
    - `test.txt`: test file
    - `kg_final.txt`: KG file

  - `Yelp_0.1/`: Preprocessed sparse dataset of Yelp
    - `train.txt`: train file
    - `valid.txt`: valid file
    - `test.txt`: test file
    - `kg_final.txt`: KG file
    
- `src/`
  - `data_preparation.py`: proprocess raw data files
  - `dataloader.py`: dataloader for training and testing
  - `models.py`: colleciton of models (Embedding Module, Logical Reasoning Module, Prediction Module) in KLQR
  - `path_extration.py`: path extraction code for Query Generation Module
  - `util.py`: collection of helper funtions
	
- `main.py`: main code
  
## Running the codes with Reproducibility

- BookCrossing 
  ```
  $ python src/data_preparation.py -dataset BookCrossing
  $ python src/path_extraction.py -dataset BookCrossing
  $ CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test --valid_steps 100 --data_path data/BookCrossing -n 128 -b 128 -d 32 -g 15 -lr 0.0001 --l2_lambda 1e-2 --geo beta --beta_mode "(800,2)" --tasks "1p.2p.3p"
  ```
  
- MovieLens1M_0.1 
  ```
  $ python src/path_extraction.py -dataset MovieLens1M_0.1 
  $ CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test --valid_steps 100 --data_path data/MovieLens1M_0.1 -n 128 -b 128 -d 32 -g 15 -lr 0.0001 --l2_lambda 1e-2 --geo beta --beta_mode "(400,2)" --tasks "1p.2p.3p"
  ```

- MovieLens1M_0.3
  ```
  $ python src/path_extraction.py -dataset MovieLens1M_0.3
  $ CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test --valid_steps 100 --data_path data/MovieLens1M_0.3 -n 128 -b 128 -d 32 -g 6 -lr 0.01 --l2_lambda 1e-2 --geo beta --beta_mode "(1600,2)" --tasks "1p.2p.3p"
  ```
  
- MovieLens1M_0.5
  ```
  $ python src/path_extraction.py -dataset MovieLens1M_0.5
  $ CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test --valid_steps 100 --data_path data/MovieLens1M_0.5 -n 128 -b 128 -d 32 -g 6 -lr 0.01 --l2_lambda 1e-2 --geo beta --beta_mode "(1600,2)" --tasks "1p.2p.3p"
  ```
  
- Yelp_0.1
  ```
  $ python src/path_extraction.py -dataset Yelp_0.1
  $ CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test --valid_steps 100 --data_path data/Yelp_0.1 -n 128 -b 128 -d 32 -g 6 -lr 0.01 --l2_lambda 1e-2 --geo beta --beta_mode "(1600,2)" --tasks "1p.2p.3p"
  ```

## Plotting

To reproduce **Figure 5**, which visualizes the relationship between query uncertainty and embedding behavior:

1. **Train and test** the model using the instructions above (make sure to include the `--do_test` flag).
2. After testing, a JSON file containing **query-wise differential entropy** will be automatically saved in the `logs/` directory.
3. Run the following command to generate the plot:

   ```bash
   python plot_entropy.py 
   ```

4. This will generate a `.png` file that plots the relationship between uncertainty and ranking performance, saved in the same directory.
