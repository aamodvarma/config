from transformers import AutoModelForCausalLM, GPT2Tokenizer , PreTrainedTokenizerFast
from pathlib import Path
import torch
import argparse
import tqdm

global FILE;
global DEVICE;
def initialize(model_name, tokenizer_name):
    model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}")
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    #tokenizer = GPT2Tokenizer.from_pretrained(f"tokenizers/{tokenizer_name}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(f"tokenizers/{tokenizer_name}")
    return model, tokenizer

def generate_helper(model, tokenizer, lengths, elements, number):
    for i in tqdm.tqdm(range(len(lengths)), desc="Lengths"):
        for word in tqdm.tqdm(elements, desc="Tokenizing words"):
            try:
                model_inputs = tokenizer(word, return_tensors="pt", padding=True)
                model_inputs.to(DEVICE)
                generate(model, tokenizer, model_inputs, lengths[i], number)
            except Exception as e:
                print(f"An error occurred for element {word} : {e}")
                continue;


def generate(model, tokenizer, model_inputs, length, number):
    generated_ids = model.generate(
        **model_inputs, 
        max_length=10,
        num_beams=number,
        no_repeat_ngram_size=2,
        top_k=50,
        # top_p=0.95,
        # temperature=0.5,
        num_return_sequences=number,
        # do_sample=True,
        pad_token_id=model.config.eos_token_id,
    )
    generated_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids]
    generated_sequences = [item + "\n" for item in generated_sequences]
    with open(f"./generated_files/{FILE}", "a") as f:
        f.writelines(generated_sequences)
    
    

def main():
    parser = argparse.ArgumentParser(
        description="Generate molecular composition"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="directory path to the trained model, to see available models type 'ls models'",
    )

    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        required=True,
        help="a valid tokenizer within tokenizer directory, to see available tokenizers type 'ls tokenizers'",
    )

    parser.add_argument(
        "-l",
        "--length",
        type=str,
        help="length of molecule you want, represent as tuple 5,6,7, default is 5,6,7,8,9,10"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="first element of genreated sequence, represent as H,He,Li,Be default is all elements in vocab.txt"
    )

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="number of generated molecules for each for the given lengths and elements, default is 100. This would imply total molecule of 5 * 88 * 100 = 44000"
    )
    args = parser.parse_args()

    global FILE
    FILE = args.output

    lengths = [5,6,7,8,9,10] if args.length == None else list(map(int, args.length.split(",")))

    with open("./vocab.txt", "r") as f:
        all_elements = f.read().split()
        if args.input == None:
            elements = all_elements
        else:
            for a in args.input.split(","):
                if a not in all_elements:
                    raise ValueError("Enter valid elements")
            elements = args.input.split(",")
            
    number = 100 if args.number == None else args.number


    model, tokenizer = initialize(args.model, args.tokenizer)

    generate_helper(model, tokenizer, lengths, elements, number)
    # model_inputs = tokenizer("Na", return_tensors="pt")
    # print(tokenizer.get_vocab())

if __name__ == "__main__":
    main();
