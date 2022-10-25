import pandas as pd
from split import split


def statistic_sentences(dataset, mode):
    data_frame = pd.read_csv("../data/%s/%s.csv" % (dataset, mode))
    all_intents = set(pd.read_csv("../data/%s/intent.csv" % dataset)['intent'])
    total_ind_num = 0
    total_ood_num = 0
    total_pure_ood_num = 0
    total_mixed_ood_num = 0
    total_intents_num = 0
    total_utters_num = 0
    for i in range(len(split[dataset])):
        valid_ood_intents = set(split[dataset][i]['valid_ood_label'].strip().split(","))
        test_ood_intents = set(split[dataset][i]['test_ood_label'].strip().split(","))
        ind_intents = all_intents - valid_ood_intents - test_ood_intents
        ood_intents = set()
        if mode == 'valid':
            ood_intents = valid_ood_intents
        elif mode == 'test':
            ood_intents = test_ood_intents
        ind_num = 0
        ood_num = 0
        mixed_ood_num = 0
        pure_ood_num = 0
        for intent_item in data_frame['intent']:
            intents = set(intent_item.strip().split("#"))
            if len(intents - ind_intents) <= 0:
                ind_num += 1
                total_intents_num += len(intents)
                total_utters_num += 1
                continue

            if len(ood_intents) > 0 and len(intents - ind_intents - ood_intents) <= 0:
                ood_num += 1
                total_intents_num += len(intents)
                total_utters_num += 1
                if len(intents - ood_intents) <= 0:
                    pure_ood_num += 1
                else:
                    mixed_ood_num += 1

        total_ind_num += ind_num
        total_ood_num += ood_num
        total_pure_ood_num += pure_ood_num
        total_mixed_ood_num += mixed_ood_num
    return {
        "ind_num": total_ind_num / len(split[dataset]),
        "ood_num": total_ood_num / len(split[dataset]),
        "pure_ood_num": total_pure_ood_num / len(split[dataset]),
        "mixed_ood_num": total_mixed_ood_num / len(split[dataset]),
        "total_intents_num": total_intents_num,
        "total_utters_num": total_utters_num,
    }


def statistic_intents(dataset):
    all_intents = set(pd.read_csv("../data/%s/intent.csv" % dataset)['intent'])
    ind_intent_num_sum = 0
    for i in range(len(split[dataset])):
        valid_ood_intents = set(split[dataset][i]['valid_ood_label'].strip().split(","))
        test_ood_intents = set(split[dataset][i]['test_ood_label'].strip().split(","))
        ind_intent_num_sum += len(all_intents - valid_ood_intents - test_ood_intents)

    return {
        "ind_intent_num": round(ind_intent_num_sum / len(split[dataset])),
    }


def statistic_all(dataset):
    train_stat = statistic_sentences(dataset, "train")
    valid_stat = statistic_sentences(dataset, "valid")
    test_stat = statistic_sentences(dataset, "test")
    intents_stat = statistic_intents(dataset)

    total_intents_num, total_utters_num = 0, 0
    for stat in [train_stat, valid_stat, test_stat]:
        total_intents_num += stat['total_intents_num']
        total_utters_num += stat['total_utters_num']

    return {
        'Train-IND': round(train_stat['ind_num']),
        'Validation-IND': round(valid_stat['ind_num']),
        'Validation-OOD': round(valid_stat['ood_num']),
        'Test-IND': round(test_stat['ind_num']),
        'Test-OOD': round(test_stat['ood_num']),
        "Test-Mixed OOD": round(test_stat['mixed_ood_num']),
        "Test-Pure OOD": round(test_stat['pure_ood_num']),
        "Number of known intents": intents_stat['ind_intent_num'],
        "Average intent number per utterance": round(total_intents_num / total_utters_num, 1),
    }


if __name__ == '__main__':
    print("mixsnips:", statistic_all("mixsnips"))
    print("multiwoz23:", statistic_all("multiwoz23"))
    print("atis:", statistic_all("atis"))
    print("FSPS:", statistic_all("FSPS"))
