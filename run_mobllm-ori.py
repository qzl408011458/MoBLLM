# import openai
import argparse
import os
import pickle
import time
import ast
import pandas as pd
import logging
from datetime import datetime

import transformers
import torch
from openai import AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

import os

os.environ["HUGGINGFACE_TOKEN"] = ""


# Helper function
def get_chat_completion(prompt):
    messages = [{"role": "user", "content": prompt}]

    response = pipeline(messages)
    res_content = response[0]['generated_text'][1]['content']
    token_usage = len(pipeline.tokenizer.tokenize(response[0]['generated_text'][1]['content'])) + \
                  len(pipeline.tokenizer.tokenize(response[0]['generated_text'][0]['content']))

    return res_content, token_usage


def convert_to_12_hour_clock(minutes):
    if minutes < 0 or minutes >= 1440:
        return "Invalid input. Minutes should be between 0 and 1439."

    hours = minutes // 60
    minutes %= 60

    period = "AM"
    if hours >= 12:
        period = "PM"

    if hours == 0:
        hours = 12
    elif hours > 12:
        hours -= 12

    return f"{hours:02d}:{minutes:02d} {period}"

def convert_to_12_hour_clock2(minutes):
    if minutes < 0 or minutes >= 1440:
        return "Invalid input. Minutes should be between 0 and 1439."

    hours = minutes // 60
    minutes %= 60

    period = "AM"
    if hours >= 12:
        period = "PM"

    # if hours == 0:
    #     hours = 12
    elif hours > 12:
        hours -= 12

    return f"{hours:02d}:{minutes:02d} {period}"


def int2dow(int_day):
    tmp = {0:'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    return tmp[int_day]


def get_logger(logger_name, log_dir='logs/'):
    # Create log dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set its log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a file handler and set its log level
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    log_file = 'log_file' + formatted_datetime + '.log'
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_user_data(train_data, uid, num_historical_stay, logger):
    user_train = train_data[train_data['user_id']==uid]
    logger.info(f"Length of user {uid} train data: {len(user_train)}")
    user_train = user_train.tail(num_historical_stay)
    logger.info(f"Number of user historical stays: {len(user_train)}")
    return user_train


# Organising data
def organise_data0(user_train, test_file, uid, logger, num_context_stay=5):
    # Use another way of organising data
    historical_data = []

    for _, row in user_train.iterrows():
        #print(f"row['start_min'] {row['start_min']}")
        historical_data.append((convert_to_12_hour_clock2(int(row['start_min'])), int2dow(row['weekday']), int(row['duration']), row['location_id']))

    logger.info(f"historical_data: {historical_data}")
    logger.info(f"Number of historical_data: {len(historical_data)}")

    # Get user ith test data
    list_user_dict = []
    for i_dict in test_file:
        if i_dict['user_X'][0] == uid:
            list_user_dict.append(i_dict)
        
    predict_X = []
    predict_y = []
    for i_dict in list_user_dict:
        # select last (num_context_stay) activities as context
        # select the next activity's partial information as target
        # the next activity's location as predict_y

        construct_dict = {}
        context = list(zip([convert_to_12_hour_clock2(int(item)) for item in i_dict['start_min_X'][-num_context_stay:]],
                           [int2dow(i) for i in i_dict['weekday_X'][-num_context_stay:]], 
                           [int(i) for i in i_dict['dur_X'][-num_context_stay:]], 
                           i_dict['X'][-num_context_stay:]))
        target = (convert_to_12_hour_clock2(int(i_dict['start_min_Y'])), int2dow(i_dict['weekday_Y']), None, "<next_place_id>")
        construct_dict['context_stay'] = context
        construct_dict['target_stay'] = target
        predict_y.append(i_dict['Y'])
        predict_X.append(construct_dict)

    #logger.info(f"predict_data: {predict_X}")
    logger.info(f"Number of predict_data: {len(predict_X)}")
    logger.info(f"predict_y: {predict_y}")
    logger.info(f"Number of predict_y: {len(predict_y)}")
    return historical_data, predict_X, predict_y


def organise_data(test_data, uid, logger, num_context_stay=5):
    # Use another way of organising data
    # historical_data = []

    data_uid = test_data[uid]
    predict_X = []
    predict_y = []

    for i in range(len(data_uid)):
        his_i, con_i, tar_i = data_uid[i]

        history, context = [], []

        for act in his_i:
            history.append((convert_to_12_hour_clock2(act[0]), int2dow(act[1]),
                            act[2], act[3], act[4]))

        for act in con_i:
            context.append((convert_to_12_hour_clock2(act[0]), int2dow(act[1]),
                            act[2], act[3], act[4]))

        target = (
            convert_to_12_hour_clock2(tar_i[0][0]),
            int2dow(tar_i[0][1]), None, tar_i[0][3], "<next_place_id>")

        construct_dict = {}
        construct_dict['history_stay'] = history
        construct_dict['context_stay'] = context
        construct_dict['target_stay'] = target

        predict_y.append(tar_i[0][4])
        predict_X.append(construct_dict)

    logger.info(f"Number of predict_data: {len(predict_X)}")
    logger.info(f"predict_y: {predict_y}")
    logger.info(f"Number of predict_y: {len(predict_y)}")
    return predict_X, predict_y


def single_query_top1(X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user have been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, getoff_id, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
    getoff_id: an integer representing the ID of the alighting station (e.g., 1,2,3,..., etc.) or home (i.e., 0), which indicates the location closest to the activity place (i.e., place_id).
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
     
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you learned from <history>, e.g., repeated visit to a certain place during certain time;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal and spatial information (i.e., start_time, day_of_week and getoff_id) of target stay, which is important because people's activity varies during different times (e.g., nighttime versus daytime),
    and on different days (e.g., weekday versus weekend), and different places (e.g., central business district versus residential district)

    The data are as follows:
    <history>: {X['history_stay']}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    
    Please organize your answer in a JSON object containing following keys: "prediction" (place ID) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.
    """
    response, usage = get_chat_completion(prompt)
    return response, usage


def single_query_top10(X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user have been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {X['history_stay']}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    response, usage = get_chat_completion(prompt)
    return response, usage


def single_query_top1_wot(X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user have been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. 
    place_id: an integer representing the unique place ID, which indicates where the stay is.    
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visit to a certain place during certain time;
    2. the context stays in <context>, which provide more recent activities of this user.

    Please organize your answer in a JSON object containing following keys: "prediction" (place ID) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {X['history_stay']}
    <context>: {X['context_stay']}
    """
    response, usage = get_chat_completion(prompt)
    return response, usage


def single_query_top10_wot(X):
    """
    Make a single query of 10 most likely places, without time information
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user have been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. 
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user.
  
    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not use line breaks in the reason.

    The data are as follows:
    <history>: {X['history_stay']}
    <context>: {X['context_stay']}
    """
    response, usage = get_chat_completion(prompt)
    return response, usage


def load_results(filename):
    # Load previously saved results from a CSV file    
    results = pd.read_csv(filename)
    return results


def single_user_query(uid, predict_X, predict_y,logger, top_k, is_wt, output_dir, sleep_query, sleep_crash):
    # Initialize variables
    global query_ind
    total_queries = len(predict_X)
    logger.info(f"Total_queries: {total_queries}")

    processed_queries = 0
    current_results = pd.DataFrame({
        'user_id': None,
        'ground_truth': None,
        'prediction': None,
        'reason': None
    }, index=[])

    out_filename = f"{uid:02d}" + ".csv"
    out_filepath = os.path.join(output_dir, out_filename)

    try:
        # Attempt to load previous results if available
        current_results = load_results(out_filepath)
        processed_queries = len(current_results)
        logger.info(f"Loaded {processed_queries} previous results.")
    except FileNotFoundError:
        logger.info("No previous results found. Starting from scratch.")


    # Process remaining queries
    for i in range(processed_queries, total_queries):
        query_ind += 1
        if is_wt is True:
            if top_k == 1:
                try:
                    logger.info(f'The {query_ind}th sample: ')
                    response, usage = single_query_top1(predict_X[i])
                except Exception as e:
                    logger.info(e)
                    logger.info(f"API request failed for the {query_ind}th query")
                    time.sleep(sleep_crash)
                    return  # Exit the loop if API crashes
            elif top_k == 10:
                response, usage = single_query_top10(predict_X[i])
            else:
                raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")
        else:
            if top_k == 1:
                response, usage = single_query_top1_wot(predict_X[i])
            elif top_k == 10:
                response, usage = single_query_top10_wot(predict_X[i])
            else:
                raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")

        # Log the prediction results and usage.
        logger.info(f"Pred results: {response}")
        logger.info(f"Ground truth: {predict_y[i]}")
        logger.info(f"total tokens: {usage}")

        # logger.info(usage)
        try:
            if '{' in response:
                response = response[response.find('{'):]
            if '}' in response:
                response = response[:response.find('}') + 1]
            res_dict = {}
            try:
                res_dict = ast.literal_eval(response)  # Convert the string to a dictionary object
            except:
                response.split(',')
                for seg in response.strip('{').strip('}').strip().split(','):
                    if 'prediction' in seg:
                        res_dict['prediction'] = seg.split(':')[-1].strip()
                    if 'reason' in seg:
                        res_dict['reason'] = seg.split(':')[-1].strip()

            if top_k != 1:
                res_dict['prediction'] = str(res_dict['prediction'])
            res_dict['user_id'] = uid
            res_dict['ground_truth'] = predict_y[i]
            if 'prediction' not in res_dict:
                res_dict['prediction'] = -100
                logger.info(f"the {query_ind}th query gets LLM hallucination\n")

        except Exception as e:
            res_dict = {'user_id': uid, 'ground_truth': predict_y[i], 'prediction': -100, 'reason': None}
            logger.info(e)
            # logger.info(f"API request failed for the {i + 1}th query\n")
            logger.info(f"the {query_ind}th query gets LLM hallucination\n")
            # logger.info('#####################################################################################')
            # time.sleep(sleep_crash)
        finally:
            new_row = pd.DataFrame(res_dict, index=[0])  # A dataframe with only one record
            current_results = pd.concat([current_results, new_row], ignore_index=True)  # Add new row to the current df


    # Save the current results
    current_results.to_csv(out_filepath, index=False)
    #save_results(current_results, out_filename)
    logger.info(f"Saved {len(current_results)} results to {out_filepath}")



def query_all_user(uid_list, logger, num_historical_stay, num_context_stay, test_data, top_k, is_wt, output_dir, sleep_query, sleep_crash):
    for uid in uid_list:
        logger.info(f"=================Processing user {uid}==================")
        # user_train = get_user_data(train_data, uid, num_historical_stay, logger)
        predict_X, predict_y = organise_data(test_data, uid, logger, num_context_stay)
        single_user_query(uid, predict_X, predict_y, logger, top_k=top_k, is_wt=is_wt, output_dir=output_dir, sleep_query=sleep_query, sleep_crash=sleep_crash)

# Get the remaning user
def get_unqueried_user(output_dir='output/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_user_id = [i for i in range(0, test_uids_count)]
    processed_id = [int(file.split('.')[0]) for file in os.listdir(output_dir) if file.endswith('.csv')]
    remain_id = [i for i in all_user_id if i not in processed_id]
    print(remain_id)
    print(f"Number of the remaining id: {len(remain_id)}")
    return remain_id



def parse_args(argv=None):
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(
        description='Run LLM for prediction')
    parser.add_argument('--data', default='ori_hk_incident', type=str,
                        help='Choose the dataset')
    parser.add_argument('--base_path', default='meta-llama/Meta-Llama-3.1-8B-Instruct', type=str,
                        help='Choose a base model path')
    parser.add_argument('--ft_path', default='modelSave/llama3_1_instruct_olora/checkpoint-11447', type=str,
                        help='Choose model path')
    parser.add_argument('--ft_name', default='olora', type=str,
                        help='Give ft model name')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='GPU')

    # parser.add_argument('--batch_size', default=64, type=int,
    #                     help='Choose batch size')
    global args
    args = parser.parse_args(argv)

def main():
    global pipeline, test_uids_count, query_ind, model_gpt
    model_gpt = 'gpt-4o-mini'
    query_ind = 0
    parse_args()
    # Parameters
    num_historical_stay = 40  # M
    num_context_stay = 5  # N
    top_k = 1  # the number of output places k
    with_time = True  # whether incorporate temporal information for target stay
    sleep_single_query = 1  # the sleep time between queries
    sleep_if_crash = 10  # the sleep time if the server crashes
    # output_dir = 'output/ft_ori/top1'  # the output path
    # log_dir = 'logs/ft_ori/top1'  # the log dir

    # load_lora = True
    # load_lora = False
    model_path = args.base_path
    lora_path = args.ft_path
    ft_name = args.ft_name
    dataname = args.data


    if args.ft_name == 'loftq':
        MODEL_ID = "LoftQ/Meta-Llama-3-8B-Instruct-4bit-64rank"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                                  use_fast=False, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,  # you may change it with different pretrain_models
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
            device_map=args.device
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_id=lora_path,
            # subfolder="loftq_init",
            is_trainable=False,
        )
        output_dir = f"output/ft_{ft_name}_{dataname}/top{top_k}"  # the output path
        log_dir = f"logs/ft_{ft_name}_{dataname}/top{top_k}"  # the log dir
    elif args.ft_name == 'qlora':
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path,  # torch_dtype=torch.bfloat16,
                                                     use_cache=False,
                                                     quantization_config=nf4_config,
                                                     device_map=args.device)
        output_dir = f"output/ft_{ft_name}_{dataname}/top{top_k}"  # the output path
        log_dir = f"logs/ft_{ft_name}_{dataname}/top{top_k}"  # the log dir

    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                     use_cache=False, device_map=args.device)
        if len(args.ft_path) != 0 and args.ft_path != model_path:
            # 加载lora权重
            model = PeftModel.from_pretrained(model, model_id=lora_path)
            output_dir = f"output/ft_{ft_name}_{dataname}/top{top_k}"  # the output path
            log_dir = f"logs/ft_{ft_name}_{dataname}/top{top_k}"  # the log dir
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_cache=False,
                                                         device_map=args.device)
            output_dir = f"output/raw_{dataname}_test/top{top_k}"  # the output path
            log_dir = f"logs/raw_{dataname}_test/top{top_k}"  # the log dir

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            # device=0,
            do_sample=False,
            max_new_tokens=300,
            eos_token_id= [128001, 128009],
            pad_token_id=0
        )
        pipeline.generation_config.temperature = None
        pipeline.generation_config.top_p = None



    if args.data == 'ori_hk':
        with open('data4eval/ori_hk.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)

    if args.data == 'ori_hz':
        with open('data4eval/ori_hz.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)

    if args.data == 'ori_hk_network_change':
        with open('data4eval/ori_network_change.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)

    if args.data == 'ori_hk_policy_intervention':
        with open('data4eval/ori_policy_intervention.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)

    if args.data == 'ori_hk_event':
        with open('data4eval/ori_original_event.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)
    if args.data == 'ori_hk_incident':
        with open('data4eval/ori_original_incident.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)

    # Get training and validation set and merge them
    test_uids_count = len(test_data)
    test_count = 0
    for _ in test_data:
        test_count += len(test_data[_])


    print("Number of total test sample: ", test_count)

    logger = get_logger('my_logger', log_dir=log_dir)

    uid_list = get_unqueried_user(output_dir=output_dir)
    print("uid_list: ", uid_list)

    query_all_user(uid_list, logger, num_historical_stay, num_context_stay,
                   test_data, output_dir=output_dir, top_k=top_k, is_wt=with_time,
                   sleep_query=sleep_single_query, sleep_crash=sleep_if_crash)

    print("Query done")


if __name__=="__main__":
    main()
