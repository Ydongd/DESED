## Event Detection

Codes about three different levels of attention mechanisms: token-level, utterance-level and hybrid. Codes here are used for FOSAED. When used for ACE05, one should slightly modify `data_processor.py`, `data_utils.py` (as in annotation) and use `const_ace.py` instead of `const.py`.

## Main APP

| Arguments                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| data_dir                 | The input data directory                                     |
| model_type               | Seletion of model type                                       |
| model_name_or_path       | Path to pretrained model or shortcut name of the model       |
| output_dir               | The output directory                                         |
| num_sentence             | Number of utterances used                                    |
| do_train                 | Whether to run training                                      |
| do_eval                  | Whether to run eval                                          |
| evaluate_during_training | Whether to run evaluation during training at each logging step |
| evaluate_after_epoch     | Whether to run evaluation after every epoch                  |
| per_gpu_train_batch_size | Batch size per GPU/CPU for training                          |
| per_gpu_eval_batch_size  | Batch size per GPU/CPU for evaluation                        |
| learning_rate            | The initial learning rate for Adam                           |

More details can be seen in `arguments.py`.