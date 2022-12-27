def model_train(dataloader , nb_epochs = 500):

    model = create_model()
    define_criterion = torch.nn.MSELoss(size_average=False)
    SGD_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(nb_epochs): 

        model.train()

        for (input, target) in tqdm(dataloader):

            input = input.to(device)
            output = output.to(device)

            output = model(input) 
            loss = define_criterion(output, target) 

            SGD_optimizer.zero_grad() 
            loss.backward() 

            SGD_optimizer.step() 
            print('epoch {}, loss function {}'.format(epoch, loss.item()))
    
    return model

def preprocessing(X, train, vectorizer_text = None, vectorizer_hashtags = None, std_clf = None):
    # We set up an Tfidf Vectorizer that will use the top 100 tokens from the tweets. We also remove stopwords.
    # To do that we have to fit our training dataset and then transform both the training and testing dataset. 
    
    X_only_int = X.select_dtypes('int')

    if train == True:
        vectorizer_text = TfidfVectorizer(max_features=150, stop_words=stopwords.words('french'))
        X_text = pd.DataFrame(vectorizer_text.fit_transform(X['text']).toarray(), index = X.index)
    else : X_text = pd.DataFrame(vectorizer_text.transform(X['text']).toarray(), index = X.index)

    X_new = pd.concat([X_only_int, X_text], axis = 1)

    X_new['month'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%m")))
    X_new['hour'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%H")))

    hashtags = X['hashtags'].apply(lambda x : x[2:-2].split(','))
    hashtags.apply(lambda x : x.remove('') if '' in x else x)
    hashtags = hashtags.apply(lambda x : len(x))
    X_new['hashtags_count'] = hashtags

    if train == True:
        hashtags, vectorizer_hashtags = preprocess_hashtags(X, train = train)
        X_new = pd.concat([X_new, hashtags], axis =1)
    else :
        hashtags, vectorizer_hashtags = preprocess_hashtags(X, train = train, vectorizer= vectorizer_hashtags)
        X_new = pd.concat([X_new, hashtags], axis =1)

    urls = X['urls'].apply(lambda x : x[2:-2].split(','))
    urls.apply(lambda x : x.remove('') if '' in x else x)
    urls = urls.apply(lambda x : len(x))
    X_new['urls_count'] = urls

    sia = SentimentIntensityAnalyzer()
    X_new['polarity_scores_neg'] = X['text'].apply(lambda text : sia.polarity_scores(text)['neg'])
    X_new['polarity_scores_pos'] = X['text'].apply(lambda text : sia.polarity_scores(text)['pos'])
    X_new['polarity_scores_compound'] = X['text'].apply(lambda text : sia.polarity_scores(text)['compound'])

    X_new = X_new.drop(['TweetID'], axis = 1)

    if train == True:
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=2))
        std_clf.fit(X_new)
        X_transformed = std_clf.transform(X_new)
        return X_new, vectorizer_text, vectorizer_hashtags, std_clf
    else: 
        X_transformed = std_clf.transform(X_new)
        return X_new, vectorizer_text, vectorizer_hashtags, std_clf