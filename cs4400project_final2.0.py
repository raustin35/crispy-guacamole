import pandas as pd
import numpy as np
from os.path import join

# 1. read data

ltable = pd.read_csv('ltable.csv')
rtable = pd.read_csv('rtable.csv')
train = pd.read_csv('train.csv')


# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR

blank_brand_l = []
blank_brand_r = []

def block_by_brand(ltable, rtable):
    # ensure brand is str
    ltable['brand'] = ltable['brand'].astype(str)
    rtable['brand'] = rtable['brand'].astype(str)

    # get all brands
    brands_l = set(ltable["brand"].values)
    brands_r = set(rtable["brand"].values)
    brands = brands_l.union(brands_r)

    # map each brand to left ids and right ids
    brand2ids_l = {b.lower(): [] for b in brands}
    brand2ids_r = {b.lower(): [] for b in brands}
    for i, x in ltable.iterrows():
        brand2ids_l[x["brand"].lower()].append(x["id"])
    for i, x in rtable.iterrows():
        brand2ids_r[x["brand"].lower()].append(x["id"])
        
    blank_brand_l = brand2ids_l["nan"]
    blank_brand_r = brand2ids_r["nan"]
    
    print(len(blank_brand_l), len(blank_brand_r))
    
#    del brand2ids_l["nan"]
#    del brand2ids_r["nan"]
#    brands.remove('nan')

    # put id pairs that share the same brand in candidate set
    candset = []
    for brd in brands:
        l_ids = brand2ids_l[brd] 
        r_ids = brand2ids_r[brd]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.append([l_ids[i], r_ids[j]])
    return candset


def block_by_modelno(ltable, rtable):
    # ensure modelno is str
    ltable['modelno'] = ltable['modelno'].astype(str)
    rtable['modelno'] = rtable['modelno'].astype(str)

    # get all modelnos
    modelnos_l = set(ltable["modelno"].values)
    modelnos_r = set(rtable["modelno"].values)
    modelnos = modelnos_l.union(modelnos_r)

    # map each modelno to left ids and right ids
    modelno2ids_l = {b.lower(): [] for b in modelnos}
    modelno2ids_r = {b.lower(): [] for b in modelnos}
    for i, x in ltable.iterrows():
        modelno2ids_l[x["modelno"].lower()].append(x["id"])
    for i, x in rtable.iterrows():
        modelno2ids_r[x["modelno"].lower()].append(x["id"])
        
    del modelno2ids_l["nan"]
    del modelno2ids_r["nan"]
    modelnos.remove('nan')

    # put id pairs that share the same modelno in candidate set
    candset = []
    for mn in modelnos:
        l_ids = modelno2ids_l[mn]
        r_ids = modelno2ids_r[mn]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.append([l_ids[i], r_ids[j]])
    return candset

# blocking to reduce the number of pairs to be compared
candset_brand = block_by_brand(ltable, rtable)
candset_modelno = block_by_modelno(ltable, rtable)
print(len(candset_brand))
print(len(candset_modelno))
#candset = (candset_brand + candset_modelno)
candset = [tup1 for tup1 in candset_brand 
       for tup2 in candset_modelno if tup1 == tup2]
print(len(candset))
  
#candset_no_modelno = candset_brand[:]
#
#for tup1 in candset_no_modelno:
#    for tup2 in candset_modelno:
#        if tup1[0] == tup2[0] and tup1[1] != tup2[1] and tup1 in candset_no_modelno:
#            candset_no_modelno.remove(tup1)
#
#candset = candset_modelno + candset_no_modelno
    
print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking",len(candset))
candset_df = pairs2LR(ltable, rtable, candset)



# 3. Feature engineering
#import Levenshtein as lev
from rapidfuzz import fuzz


def jaccard_similarity(row, attr):
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))


def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    #return lev.distance(x, y)
    return (100 - fuzz.ratio(x,y))

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "brand", "modelno", "price"]
    features = []
    for attr in attrs:
        j_sim = LR.apply(jaccard_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)
    features = np.array(features).T
    return features
candset_features = feature_engineering(candset_df)

# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values

# 4. Model training and prediction
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight="balanced", random_state=0)
rf.fit(training_features, training_label)
y_pred = rf.predict(candset_features)

# 5. output

matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)