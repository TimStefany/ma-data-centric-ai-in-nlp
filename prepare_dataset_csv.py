import json
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



parser = argparse.ArgumentParser(
    description="Dataset transformation and splitting")

parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-d',
                        dest="dataset_name",
                        help="name of the dataset to use. should be equal to the name of the directory containing the raw_data.csv inside data.",
                        type=str,
                        required=True)
optional = parser.add_argument_group('optional arguments')
optional.add_argument('-tr',
                        dest="train_split",
                        help="float number defining size of train split",
                        default=0.9,
                        type=float)
optional.add_argument('-vl',
                        dest="val_split",
                        help="float number defining size of validation split",
                        default=0.1,
                        type=float)
optional.add_argument('-te',
                        dest="test_split",
                        help="float number defining size of test split",
                        default=0.1,
                        type=float)

if __name__ == "__main__":
    # get arguments
    args = parser.parse_args()
    # ensure train and test sum up to 1
    if args.train_split + args.test_split != 1:
        raise ValueError("train and test splits should add up to one")

    # import and transform data
    directory = f'data/{args.dataset_name}/'
    df_raw = pd.read_csv(directory + 'raw_data.csv')
    # check which description columns are present
    if 'description_startupdetector' and 'startup_description' in df_raw.columns:
        # merge both description columns by filling empty values in description_startupdetector
        df_raw['description_startupdetector'] = df_raw['description_startupdetector'].fillna(df_raw['startup_description'])
        # rename column to description
        df_raw = df_raw.rename(columns={'description_startupdetector': 'description'})
    elif 'description' in df_raw.columns:
        print('description column found. No merge for description necessary.')
    else:
        print('no valid description column/columns found.\n'
              'csv file needs to contain either "description" or "description_startupdetector" and "startup_description".')
        SystemExit()

    # transform labels to numeric
    le = LabelEncoder()
    df_raw['labels'] = le.fit_transform(df_raw['industry'])

    # drop unecessary columns
    if 'startup_description' in df_raw.columns:
        df_raw = df_raw.drop(columns=['startup_description'])

    # save as full csv
    df_raw.to_csv(directory + 'dataset_full.csv')
    df_raw.info()

    # split dataset according to train, val and test splits defined in arguments
    df = df_raw[['description', 'labels']]
    train, test = train_test_split(df, test_size=args.test_split)
    train, val = train_test_split(train, test_size=args.val_split)

    # save train val and test as seperate csv files
    train.to_csv(directory + 'train.csv', index=False)
    val.to_csv(directory + 'validation.csv', index=False)
    test.to_csv(directory + 'test.csv', index=False)

    # write info to info.json
    label_text_map = dict(zip(le.classes_, le.transform(le.classes_)))
    for keys in label_text_map:
        label_text_map[keys] = int(label_text_map[keys])
    info = {
        "num_labels": len(df_raw['industry'].unique()),
        "total_size": len(df_raw),
        "train_size": len(train),
        "train_chunk": args.train_split,
        "validation_size": len(val),
        "validation_chunk": args.val_split,
        "test": len(test),
        "test_chunk": args.test_split,
        "label_text_map": label_text_map
    }
    json_object = json.dumps(info, indent=4)
    with open(directory + "info.json", "w") as outfile:
        outfile.write(json_object)
