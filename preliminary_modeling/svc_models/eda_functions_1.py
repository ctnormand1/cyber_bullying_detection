import pandas as pd
import warnings


def merge_with_target(df_comments, df_annotations, target_col_name,
threshold=0.5):
    """Creates a dataframe with the fields we need for modeling.

    Args:
        df_comments -- dataframe created from the *_annotated_comments file
        df_annotations -- dataframe created from the *_annotations file
        target_col_name -- name of the target column in df_annotations
        threshold -- target = 1 if the proportion of reviewers who flag the
        comment is >= threshold. Default is 0.5.

    Returns:
        Dataframe with columns = [rev_id, comment, target]
    """
    # target = 1 if proportion of annotators who flagged the comment is
    # >= threshold.
    target = (df_annotations.groupby('rev_id')[target_col_name].mean() >= \
    threshold).astype(int).rename('target')

    # merge and return
    return df_comments[['rev_id', 'comment']].merge(target,
    left_on='rev_id', right_index=True).set_index('rev_id')


def split_data(data, pct_positive, test_size=None, train_size=None,
random_state=None):
    """
    A custom train_test_split implementation that creates a training dataset
    with a specific proportion of positive classes. This is meant to
    combat class imbalance in training data ONLY. Testing data will retain the
    original class proportions.

    Args:
        data -- the dataframe acquired from merge_with_target()
        pct_positive -- desired proportion of positive classes in training data.
            (float between 0 and 1)
        test_size -- number of samples to include in the testing dataset.
            (integer, default None --> 20% of the full dataset)
        train_size -- number of samples to include in the training dataset.
            (integer, default None --> maximum training size for given
            parameters)
    """
    # get test data
    if test_size:
        num_test_samples = int(test_size)
    else:
        num_test_samples = int(0.3 * data.shape[0])

    df_test = data.sample(num_test_samples, random_state=random_state)
    X_test = df_test['comment']
    y_test = df_test['target']

    # get train data
    # separate positive and negative classes
    df_train = data.drop(df_test.index)
    df_train_pos = df_train[df_train['target'] == 1]
    df_train_neg = df_train[df_train['target'] == 0]

    # some input validation for train_size
    max_train_size = int(df_train_pos.shape[0] / pct_positive)
    if not train_size:
        train_size = max_train_size
    elif train_size > max_train_size:
        warnings.warn(f'train_size of {train_size} exceeds the amount of '
        'training data available for the given parameters. '
        f'Resetting train size to {max_train_size}')
        train_size = max_train_size
    else:
        train_size = int(train_size)

    # assemble train dataframe
    num_pos_samples = int(pct_positive * train_size)
    num_neg_samples = train_size - num_pos_samples

    df_train = pd.concat([
        df_train_pos.sample(num_pos_samples, random_state=random_state),
        df_train_neg.sample(num_neg_samples)]).sample(
        frac=1, random_state=random_state)

    X_train = df_train['comment']
    y_train = df_train['target']

    return (X_train, X_test, y_train, y_test)
