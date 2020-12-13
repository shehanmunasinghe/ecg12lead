def filter_metadata(metadata,use_labels=None,use_datasets=None):
    # TODO - use_labels - remove all zero labels; consider use_datasets
    if use_datasets:
        # use_datasets = ['CPSC', 'CPSC-Extra','StPetersburg', 'PTB', 'PTB-XL', 'Georgia']
        metadata = metadata.query("_dataset == @use_datasets")

    metadata = metadata[
        ['_dataset','_filename','_split_no']+use_labels
    ]

    query_str = '|'.join(["%s !=0 "%lbl for lbl in use_labels])
    metadata = metadata.query(query_str)

    count = metadata[use_labels].sum().to_dict()
    print(count)

    return metadata