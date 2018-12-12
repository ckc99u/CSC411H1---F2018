#
# CSC411H1 Fall 2018 HW1
# Lino Lastella 1001237654
#

import numpy as np


def load_data():
    """
    loads the data, preprocesses it using a vectorizer
    (http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text),
    and splits the entire dataset randomly into 70% training, 15% validation, and 15% test examples.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    # Load the data

    with open("clean_real.txt", 'r') as RealNews:
        RealStrAr = RealNews.read().split('\n')

    with open("clean_fake.txt", 'r') as FakeNews:
        FakeStrAr = FakeNews.read().split('\n')

    # Preprocess it using a vectorizer

    MyCoolVectorizer = CountVectorizer()
    X = MyCoolVectorizer.fit_transform(RealStrAr + FakeStrAr)

    RealLabels = np.ones((len(RealStrAr), 1))  # means real
    FakeLabels = np.zeros((len(FakeStrAr), 1)) # means fake
    AllLabels = np.append(RealLabels, FakeLabels, axis=0)

    FinalTensor = np.append(X.toarray(), AllLabels, axis=1)

    # Randomize it and split it

    np.random.shuffle(FinalTensor)

    # divide and multiply by 2 just to make sure it's even
    ROUGHLY70 = 2 * ((FinalTensor.shape[0] * 70 / 100) / 2)
    ROUGHLY15 = (FinalTensor.shape[0] - ROUGHLY70) / 2

    #              TEST SET                   VALIDATION SET                      TRAINING SET                      DICTIONARY
    return (FinalTensor[:ROUGHLY15], FinalTensor[ROUGHLY15 : 2 * ROUGHLY15], FinalTensor[-ROUGHLY70:], MyCoolVectorizer.get_feature_names())


#
# global variable
#

AllSets = load_data()


def select_model():
    """
    trains the decision tree classifier using at least 5 different values of
    'max_depth', as well as two different split criteria
    (information gain and Gini coefficient), evaluates the performance of each
    one on the validation set, and prints the resulting accuracies of each model.
    You should use DecisionTreeClassifier, but you should write the validation code yourself
    """
    from sklearn import tree
    import graphviz

    ValidationSetAndLabels = AllSets[1]
    ValLabels = ValidationSetAndLabels[:, [-1]] # extract labels (last column)
    ValSet = np.delete(ValidationSetAndLabels, -1, axis=1) # delete labels

    TrainingSetAndLabels = AllSets[2]
    TrainLabels = TrainingSetAndLabels[:, [-1]] # extract labels (last column)
    TrainSet = np.delete(TrainingSetAndLabels, -1, axis=1) # delete labels

    """
    This is the code to select the best hyperparameter (part b)

    for SplitCriterion in ['entropy', 'gini']:
        print "Criterion: " + SplitCriterion + '\n'

        for MaxDepth in [int(depth) for depth in np.linspace(1, np.log2(TrainSet.shape[1]), 5)]:
            print "max_depth: " + str(MaxDepth) + '\n'

            MyTree = tree.DecisionTreeClassifier(criterion=SplitCriterion, max_depth=MaxDepth)
            MyTree = MyTree.fit(TrainSet, TrainLabels)

            Predictions = MyTree.predict(ValSet)
            Result = np.abs(Predictions - ValLabels.flatten())

            Accuracy = 100 * float(np.count_nonzero(Result == 0)) / Predictions.shape[0]

            print "Accuracy for this test is: %f %%" %Accuracy
            print '\n'

        print '\n'
    """

    MyTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=12)

    MyTree = MyTree.fit(TrainSet, TrainLabels)

    Predictions = MyTree.predict(ValSet)
    Result = np.abs(Predictions - ValLabels.flatten())

    Accuracy = 100 * float(np.count_nonzero(Result == 0)) / Predictions.shape[0]

    dot_data = tree.export_graphviz(MyTree, out_file=None, max_depth=2,
        feature_names=AllSets[3], filled=True, rounded=True, special_characters=True,
        class_names=TrainLabels.flatten().astype(str))
    graph = graphviz.Source(dot_data)
    graph.render("output")


#helper function
def H(Y):
    p_real = float(np.count_nonzero(Y.flatten() == 0)) / Y.shape[0]
    if p_real < 1 and p_real > 0:
        return  - (p_real * np.log2(p_real) + (1 - p_real) * np.log2(1 - p_real))
    return 0


def compute_information_gain(Y, xi):
    """
    computes the information gain of a split on the training data. That is,
    compute I(Y, xi), where Y is the random variable signifying whether the
    headline is real or fake, and xi is the keyword chosen for the split.
    """
    H_Y = H(Y)

    TrainSet = np.delete(AllSets[2], -1, axis=1)
    ColumnInd = AllSets[3].index(xi) # extract from dictionary

    NumHeadlines = AllSets[2].shape[0]
    AllOccurences, Count = np.unique(AllSets[2][:, ColumnInd], return_counts=True)

    TotalH_YGivenX = 0
    for i, count in zip(AllOccurences, Count):
        NewY = Y[TrainSet[:, ColumnInd] == i]

        TotalH_YGivenX +=  H(NewY) * float(count) / NumHeadlines

    return H_Y - TotalH_YGivenX


print "%f" %compute_information_gain(AllSets[2][:, [-1]], "the")
print "%f" %compute_information_gain(AllSets[2][:, [-1]], "and")
print "%f" %compute_information_gain(AllSets[2][:, [-1]], "donald")
print "%f" %compute_information_gain(AllSets[2][:, [-1]], "trumps")
print "%f" %compute_information_gain(AllSets[2][:, [-1]], "fame")
print "%f" %compute_information_gain(AllSets[2][:, [-1]], "hillary")
print "%f" %compute_information_gain(AllSets[2][:, [-1]], "sledgehammer")
