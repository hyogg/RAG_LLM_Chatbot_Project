def save_data(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(f'../files/{filename}.csv', encoding='utf-8-sig')
    df.to_json(f'../files/{filename}.json', force_ascii=False, orient='index', indent=4)
    df.to_json(f"../files/{filename}.jsonl", orient='records', lines=True, force_ascii=False)

def save_log_data(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(f'../files/{filename}.csv', encoding='utf-8-sig')