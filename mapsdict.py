
class GET_DICT():
    import pandas as pd
    data=pd.read_csv("NER.csv",encoding='unicode_escape')
    print(data.head())
    def get_dict_map(data,token_or_tag):
        token2index={}
        index2token={}
        if token_or_tag=='token':
            vocab=list(set(data['Word'].to_list()))
        else:
            vocab=list(set(data['Tag'].to_list()))
        index2token = {idx:tok for  idx, tok in enumerate(vocab)}
        token2index = {tok:idx for  idx, tok in enumerate(vocab)}
        return token2index, index2token
    
    token2idx,idx2token=get_dict_map(data,'token')
    tag2idx,idx2token=get_dict_map(data,'tag')
    data['Word_idx'] = data['Word'].map(token2idx)
    data['Tag_idx'] = data['Tag'].map(tag2idx)
    data_fillna = data.fillna(method='ffill', axis=0)
    data_group = data_fillna.groupby(['Sentence #'],as_index=False)[['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx']].agg(lambda x: list(x))

    def get_pad_train_test_val(data_group,data,tag2idx):
        from sklearn.model_selection import train_test_split
        from keras.preprocessing.sequence import pad_sequences
        from keras.utils import to_categorical
        n_token = len(list(set(data['Word'].to_list())))
        n_tag= len(list(set(data['Tag'].to_list())))
        tokens = data_group['Word_idx'].tolist()
        maxlen = max([len(s) for s in tokens])
        pad_tokens = pad_sequences(tokens, maxlen=maxlen, dtype='int32', padding='post', value= n_token - 1)
        tags=data_group['Tag_idx'].to_list()
        pad_tags = pad_sequences(tags, maxlen=maxlen, dtype='int32', padding='post', value= tag2idx["O"])
        n_tags = len(tag2idx)
        pad_tags = [to_categorical(i,num_classes=n_tags)for i in pad_tags]

        tokens_, test_tokens, tags_, test_tags = train_test_split(pad_tokens, pad_tags, test_size=0.1, train_size=0.9, random_state=2020)
        train_tokens, val_tokens, train_tags, val_tags = train_test_split(tokens_,tags_,test_size = 0.25,train_size =0.75, random_state=2020)

        print(
            'train_tokens length:', len(train_tokens),
            '\ntrain_tokens length:', len(train_tokens),
            '\ntest_tokens length:', len(test_tokens),
            '\ntest_tags:', len(test_tags),
            '\nval_tokens:', len(val_tokens),
            '\nval_tags:', len(val_tags),
        )
    
        return train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags

    train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = get_pad_train_test_val(data_group,data,tag2idx)




