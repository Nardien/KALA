import argparse
import os
import datetime

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset/NewsQA")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--train_batch_size", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=12)

    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--max_mention_length", type=int, default=30)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--version_2_with_negative", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./save/tmp")
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--verbose_logging", action="store_true")
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--read_data", action="store_true",
                        help="read data from json file")

    parser.add_argument("--kala_learning_rate", type=float, default=3e-5)
    parser.add_argument("--loc_layer", type=str, default="11")
    parser.add_argument("--domain", type=str, default="NewsQA")
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    args = parser.parse_args()

    args.do_lower_case = True

    args.output_dir = set_output_dir(args)
    print(f"Output Directory: {args.output_dir}")

    args.loc_layer = [int(x) for x in args.loc_layer.split(',')]
    args.data_dir = args.data_dir.replace("NewsQA", args.domain)
    print(f"Data Directory: {args.data_dir}")

    args.pickle_folder = args.data_dir
    print(f"KFM location: {args.loc_layer}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    return args

def set_output_dir(args):
    basedir = "./save"
    today = datetime.datetime.now().strftime("%Y%m%d")
    assert type(args.loc_layer) == str
    output_dir = f"{args.domain}"
    output_dir += f"_seed{args.seed}"
    return os.path.join(basedir, today, output_dir)