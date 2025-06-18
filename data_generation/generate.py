import json
import os
import sys
from pathlib import Path

import time
import concurrent.futures

import openai
from openai import OpenAI
import shortuuid
import tqdm

import argparse
import random

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

### old code for vLLM #####################################
# # Modify OpenAI's API key and API base to use vLLM's API server.
# openai.api_key = "EMPTY"
# openai.api_base = "http://localhost:8000/v1"

# api_base_pool = []

# # List models API
# for i in range(10):
#     openai.api_base = "http://localhost:800{}/v1".format(i)
#     try:     
#         models = openai.Model.list()["data"][0]["id"]
#         print(openai.api_base, models)
#         api_base_pool.append(openai.api_base)
#     except:
#         break

# print("API base pool: ", api_base_pool)

# sys.exit(0)
########################################################

### new code for vLLM #####################################
api_base_pool = []
for port in range(8000, 8002):
    print(f"Checking port {port}")
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1",
            timeout=1
        )
        resp = client.models.list()
        if resp.data:                            # model already loaded
            api_base_pool.append(client.base_url)
            print(client.base_url, [m.id for m in resp.data])
    except Exception:
        continue
    
print("API base pool:", api_base_pool)
# sys.exit(0)
###########################################################

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--num_threads", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--chat", action="store_true")
args = parser.parse_args()

# Assuming the ShareGPT format
data = json.load(open(args.data_path, "r"))



def generate_data(messages, idx):
    """
    Generate one sample and append it to the output file.
    Each worker builds its own OpenAI client pointing at one of the vLLM bases.
    """
    base_url = api_base_pool[idx % len(api_base_pool)]
    client = OpenAI(api_key="EMPTY", base_url=base_url)

    try:
        model_name = client.models.list().data[0].id
    except Exception:
        print(f"[{base_url}] could not fetch model list"); return

    # --------------------------- chat mode ----------------------------------
    if args.chat:
        converted = []
        output_conv = []

        # optional leading system message
        if messages and messages[0]["from"] == "system":
            converted.append({"role": "system", "content": messages[0]["text"]})
            output_conv.append(messages[0])
            messages = messages[1:]

        for m in messages[::2]:                      # user turns only
            if m["from"] != "human":
                return
            converted.append({"role": "user", "content": m["value"]})

            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=converted,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            except Exception:
                break

            choice = resp.choices[0]
            if choice.finish_reason == "length":
                break

            assistant_answer = choice.message.content.strip()
            output_conv.extend([m, {"from": "gpt", "value": assistant_answer}])
            converted.append({"role": "assistant", "content": assistant_answer})

        if not output_conv:
            return

        with open(args.output_path, "a") as f:
            f.write(json.dumps({"conversations": output_conv}) + "\n")
        return

    # ------------------------- completion mode ------------------------------
    conv = get_conversation_template(model_name)

    if messages[0]["from"] == "system":
        conv.system_message = messages[0]["text"]
        messages = messages[1:]

    conv.append_message(conv.roles[0], messages[0]["value"])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    try:
        resp = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            extra_body={
                "ignore_eos": True,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
            },
        )
    except Exception as e:
        print(e, "\n", prompt, "\nFailed to generate data")
        return
    finally:
        # force-close connection pool so sockets/FDS are returned early
        client.close()

    answer = resp.choices[0].text.strip()
    with open(args.output_path, "a") as f:
        f.write(json.dumps({"text": prompt + answer}) + "\n")
        
# ---------------------------------------------------------------------------
# 4. resume support: skip lines that already exist in output
# ---------------------------------------------------------------------------

start_idx = 0
file_path = Path("/home/nxclab/wonjun/Medusa/ShareGPT_Vicuna_unfiltered/train_shareGPT_llama3.2_1B.jsonl")
if not file_path.exists():
    file_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    file_path.touch()  # create the file
    
if Path(args.output_path).exists():
    with open(args.output_path) as f:
        start_idx = sum(1 for _ in f)
    print(f"Skip first {start_idx} samples (already generated)")

# ---------------------------------------------------------------------------
# 5. parallel generation
# ---------------------------------------------------------------------------

with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as ex:
    futures = [
        ex.submit(generate_data, sample["conversations"], i + start_idx)
        for i, sample in enumerate(data[start_idx:])
    ]
    for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass

# def generate_data(messages, idx):
#     try:
#         # load balanced
#         openai.api_base = api_base_pool[idx % len(api_base_pool)]
#         model_name=openai.Model.list()["data"][0]["id"]
        
#         print(f"Model Name: {model_name}")
#         # sys.exit(0)

#         if args.chat:
#             converted_messages = []
#             output_messages = []
#             if messages[0]["from"] == "system":
#                 converted_messages.append(
#                     {
#                         "role": "system",
#                         "content": messages[0]["text"],
#                     }
#                 )
#                 output_messages.append(messages[0])
#                 messages = messages[1:]
#             for message in messages[::2]:
#                 if message["from"] != "human":
#                     return
#                 converted_messages.append(
#                     {
#                         "role": "user",
#                         "content": message["value"],
#                     }
#                 )
#                 try:
#                     response = openai.ChatCompletion.create(
#                         model=model_name,
#                         messages=converted_messages,
#                         max_tokens=args.max_tokens,
#                         temperature=args.temperature,
#                     )
#                     if response.choices[0]['finish_reason'] == "length":
#                         break
#                     response = response.choices[0]['message']['content'].strip()
#                     output_messages.append(message)
#                     output_messages.append(
#                         {
#                             "from": "gpt",
#                             "value": response,
#                         }
#                     )
#                     converted_messages.append(
#                         {
#                             "role": "assistant",
#                             "content": response,
#                         }
#                     )
#                 except:
#                     break
#             if len(output_messages) == 0:
#                 return
#             with open(args.output_path, "a") as f:
#                 # write in share gpt format
#                 f.write(json.dumps({"conversations": output_messages}) + "\n")
#         else:
#             conv = get_conversation_template(model_name)
#             if messages[0]["from"] == "system":
#                 conv.system_message = messages[0]["text"]
#                 messages = messages[1:]
#             conv.append_message(conv.roles[0], messages[0]["value"])
#             conv.append_message(conv.roles[1], None)
#             prompt = conv.get_prompt()

#             response = openai.Completion.create(
#                 model=model_name,
#                 prompt=prompt,
#                 max_tokens=args.max_tokens,
#                 temperature=args.temperature,
#                 ignore_eos=True,
#                 skip_special_tokens=False,
#                 spaces_between_special_tokens=False,
#             )
#             response = response.choices[0]['text'].strip()
#             with open(args.output_path, "a") as f:
#                 # write in share gpt format
#                 f.write(json.dumps({"text": prompt+response}) + "\n")
#     except Exception as e:
#         print(e)
#         print(prompt)
#         print("Failed to generate data")

# # if output_path exists, count the number of lines and skip the first n data
# start = 0
# if os.path.exists(args.output_path):
#     with open(args.output_path, "r") as f:
#         start = len(f.readlines())
#         print("Skip first {} data".format(start))

# with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
#         futures = []
#         for idx, sample in enumerate(data[start:]):
#             future = executor.submit(
#                 generate_data,
#                 sample["conversations"],
#                 idx,
#             )
#             futures.append(future)

#         for future in tqdm.tqdm(
#             concurrent.futures.as_completed(futures), total=len(futures)
#         ):
#             future.result()