import pickle 
import pandas as pd
import numpy as np



def data_magic(g_a_d, g_a_n, g_n_d, g_n_n, d_a_d, d_a_n, d_n_d, d_n_n):
    df_gt= df_ngt = df_gt_na= df_ngt_na= df_gt_a= df_ngt_a= df_a=df_na= df_dec = df_ndec =df_dec_gt= df_dec_ngt = df_dec_na = df_ndec_na = df_dec_a = df_ndec_a = pd.DataFrame()

    df_gt = pd.concat([df_gt, g_a_d, g_n_d, g_a_n, g_n_n], axis=0)
    df_ngt = pd.concat([df_ngt, d_a_d, d_n_d, d_a_n, d_n_n], axis=0)

    dfs_gt = [df_gt, df_ngt]

    df_gt_na = pd.concat([df_gt_na, g_n_d, g_n_n], axis=0)
    df_ngt_na = pd.concat([df_ngt_na, d_n_d, d_n_n], axis=0)

    dfs_gt_na= [df_gt_na, df_ngt_na]

    df_gt_a = pd.concat([df_gt_a, g_a_d, g_a_n], axis=0)
    df_ngt_a = pd.concat([df_ngt_a, d_a_d, d_a_n], axis=0)

    dfs_gt_a= [df_gt_a, df_ngt_a]

    df_a = pd.concat([df_a, g_a_d, g_a_n, d_a_d, d_a_n], axis=0)
    df_na = pd.concat([df_na, g_n_d, g_n_n, d_n_d, d_n_n], axis=0)

    dfs_a = [df_a, df_na]

    dfs_a_gt = [df_gt_a, df_gt_na]

    dfs_a_ngt = [df_ngt_a, df_ngt_na]


    df_dec = pd.concat([df_dec, g_a_d, g_n_d, d_a_d, d_n_d], axis=0)
    df_ndec= pd.concat([df_ndec, g_a_n, g_n_n, d_a_n, d_n_n], axis=0)

    dfs_dec = [df_dec, df_ndec]

    df_dec_ngt =  pd.concat([df_dec_ngt, d_a_d, d_n_d], axis=0)
    df_ndec_ngt =  pd.concat([df_ndec_ngt, d_a_n, d_n_n], axis=0)

    dfs_dec_ngt = [df_dec_ngt, df_ndec_ngt]

    df_dec_gt =  pd.concat([df_dec_gt, g_a_d, g_n_d], axis=0)
    df_ndec_gt =  pd.concat([df_ndec_gt, g_a_n, g_n_n], axis=0)

    dfs_dec_gt = [df_dec_gt, df_ndec_gt]

    df_dec_na =  pd.concat([df_dec_na, g_n_d,  d_n_d], axis=0)
    df_ndec_na =  pd.concat([df_ndec_na, g_n_n, d_n_n], axis=0)

    dfs_dec_na = [df_dec_na, df_ndec_na]

    df_dec_a =  pd.concat([df_dec_a, g_a_d, d_a_d], axis=0)
    df_ndec_a =  pd.concat([df_ndec_a, g_a_n, d_a_n], axis=0)

    dfs_dec_a = [df_dec_a, df_ndec_a]

    all_dfs_combi = [dfs_gt, dfs_gt_na, dfs_gt_a, dfs_a, dfs_a_gt, dfs_a_gt, dfs_a_ngt,dfs_dec, dfs_dec_ngt, dfs_dec_gt, dfs_dec_na, dfs_dec_a ]

    with open("all_dfs_combi.pkl", "wb") as f:
        pickle.dump(all_dfs_combi, f)



if __name__ == '__main__':
    # filename = ("/gpfs/home4/ltiyavorabu/abm/basic/"+classifier)
    with open('/gpfs/home4/ltiyavorabu/abm/basic/all_data.pkl', "rb") as f:
        g_a_d, g_a_n, g_n_d, g_n_n, d_a_d, d_a_n, d_n_d, d_n_n = pickle.load(f) 
    data_magic(g_a_d, g_a_n, g_n_d, g_n_n, d_a_d, d_a_n, d_n_d, d_n_n)
