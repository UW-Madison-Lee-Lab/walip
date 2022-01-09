



def combine_files(data_name, langs, mode):
    vocabs = load_data_from_two_files(data_name, langs, mode)
    fpath = configs.paths['txt_dir'] + f'{data_name}_{langs[0]}_{langs[1]}_{mode}.txt'
    f= open(fpath, "w") 
    for i in range(len(vocabs[langs[0]])):
        f.write(f"{vocabs[langs[0]][i]}{configs.delimiters[data_name]}{vocabs[langs[1]][i]}\n")
    f.close()
    print('Done combining')


if __name__ == 'main':
