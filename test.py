
from ast import Break
from os import sep
from tracemalloc import start
from data_loader import DATA
from data_loader import DATA_RAW
from model import MODEL
import torch
import argparse
import numpy as np
import utils
from sklearn import metrics
import pandas as pd
import pickle
from scipy.stats import pearsonr
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=200, help='number of iterations')
    parser.add_argument('--decay_epoch', type=int, default=20, help='number of iterations')
    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='learning rate decay')
    parser.add_argument('--final_lr', type=float, default=1E-5,
                        help='learning rate will not decrease after hitting this threshold')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

    dataset = 'errex'


    if dataset == 'errex':
        parser.add_argument('--batch_size', type=int, default=10, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=50, help='memory size')
        parser.add_argument('--n_question', type=int, default=4, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=350, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='/content/pytorch_dkvmn/data/errex', help='data directory')
        parser.add_argument('--data_name', type=str, default='train.csv', help='data set name')
        parser.add_argument('--load', type=str, default='load.pth', help='model file to load')
        parser.add_argument('--save', type=str, default='/content/pytorch_dkvmn/save/save.pt', help='path to save model')

    
    with open('/home/thales/pytorch_dkvmn/errex_dropped.csv','rb') as file:
        df = pd.read_csv(file)
    
    with open('/home/thales/pytorch_dkvmn/train_skill_data_right.pickle','rb') as file:
        train_skill_right = pickle.load(file)['train_skill']


    with open('/home/thales/pytorch_dkvmn/ErrEx posttest data.xlsx','rb') as file:
        df_2 = pd.read_excel(file)



    params = parser.parse_args()
    params.lr = params.init_lr
    params.memory_key_state_dim = params.q_embed_dim
    params.memory_value_state_dim = params.qa_embed_dim
    params.memory_key_state_dim = params.q_embed_dim
    params.memory_value_state_dim = params.qa_embed_dim

    model = MODEL(n_question=params.n_question,
                  batch_size=params.batch_size,
                  q_embed_dim=params.q_embed_dim,
                  qa_embed_dim=params.qa_embed_dim,
                  memory_size=params.memory_size,
                  memory_key_state_dim=params.memory_key_state_dim,
                  memory_value_state_dim=params.memory_value_state_dim,
                  final_fc_dim=params.final_fc_dim)

    model.load_state_dict(torch.load('/home/thales/pytorch_dkvmn/save300.pt'))
    students = df['student_id'].unique()


    data_path = '/home/thales/pytorch_dkvmn/data/errex/train.csv'
    dat =  DATA(n_question = params.n_question,seqlen=params.seqlen,separate_char=',') 
    

    all_data = dat.load_data(data_path)
    

    train_q_data, train_qa_data, _,_ = all_data
    
    target_list = []
    pred_list = []
    batch_size=1
    
    N = train_q_data.shape[0]

    model.eval()
    all_auc = []
    all_mean_of_correctness = {}

    
    with torch.no_grad():
        start = time.time()


        for idx in range(N):

            mean_of_correctness = {}
            q_one_seq = train_q_data[idx * batch_size:(idx + 1) * batch_size, :]
            qa_batch_seq = train_qa_data[idx * batch_size:(idx + 1) * batch_size, :]
            target = train_qa_data[idx * batch_size:(idx + 1) * batch_size, :]

            target = (target-1) /params.n_question
            target = np.floor(target)

            input_q = utils.varible(torch.LongTensor(q_one_seq), params.gpu)
            input_qa = utils.varible(torch.LongTensor(qa_batch_seq), params.gpu)
            target = utils.varible(torch.FloatTensor(target), params.gpu)

            target_to_1d = torch.chunk(target, batch_size, 0)
            target_1d = torch.cat([target_to_1d[i] for i in range(batch_size)], 1)
            target_1d = target_1d.permute(1, 0)
            
            
            _, filtered_pred, filtered_target = model.forward(input_q, input_qa, target_1d)
            right_target = np.asarray(filtered_target.data.tolist())
            right_pred = np.asarray(filtered_pred.data.tolist())
            
            '''               
            index_ordDecimals = np.where(input_q[0].cpu().detach().numpy()==1)[0]
            index_placeNumber = np.where(input_q[0].cpu().detach().numpy()==2)[0]
            index_completeSeque = np.where(input_q[0].cpu().detach().numpy() ==3)[0]
            index_decimaAddition = np.where(input_q[0].cpu().detach().numpy()==4)[0]
            '''

            '''
            print(input_q[0][index_ordDecimals])
            print(right_pred[index_ordDecimals])
            print(target[0][index_ordDecimals])
            print(target)
            break
     
            mean_of_correctness['OrderingDecimals'] = np.mean(right_pred[index_ordDecimals])
            mean_of_correctness['PlacementOnNumberLine'] = np.mean(right_pred[index_placeNumber])
            mean_of_correctness['CompleteTheSequence'] = np.mean(right_pred[index_completeSeque])
            mean_of_correctness['DecimalAddition'] = np.mean(right_pred[index_decimaAddition])
            all_mean_of_correctness[students[idx]] = mean_of_correctness
            '''
        end = time.time()
    
    print("time elapsed:",end-start)



    '''
    ord_decimals = [i['OrderingDecimals'] for i in  list(all_mean_of_correctness.values())]
    place_number = [i['PlacementOnNumberLine'] for i in  list(all_mean_of_correctness.values())]
    complet_sequence = [i['CompleteTheSequence'] for i in  list(all_mean_of_correctness.values())]
    decimal_addition =  [i['DecimalAddition'] for i in  list(all_mean_of_correctness.values())]
    


    df_2 = df_2.drop([0,1], axis=0)
    df_2 = df_2.drop(df_2.columns[1:5],axis=1)


    df_post_dropped = df_2.drop_duplicates('Anon Student Id',keep='last')
    df_post_dropped = df_post_dropped.drop(df_post_dropped.index[598])

    decimal_addition_post = df_post_dropped['Unnamed: 171'].values
    ordering_decimals_post=  df_post_dropped['Unnamed: 172'].values 
    complete_sequence_post = df_post_dropped['Unnamed: 173'].values 
    placement_number_post = df_post_dropped['Unnamed: 174'].values 


    pearson_correlations = {}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

  
    pearson_correlations['OrderingDecimals'] = pearsonr(ord_decimals,ordering_decimals_post)[0]
    pearson_correlations['PlacementOnNumberLine'] = pearsonr(place_number,placement_number_post)[0]
    pearson_correlations['CompleteTheSequence'] = pearsonr(complet_sequence,complete_sequence_post)[0]
    pearson_correlations['DecimalAddition'] = pearsonr(decimal_addition,decimal_addition_post)[0]

    print(pearson_correlations)
    '''



if __name__ == "__main__":
    main()
