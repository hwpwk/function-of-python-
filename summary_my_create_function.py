def basic_check(df):
    '''
    関数内容
    読み込んだデータフレームの
    ・行と列の数
    ・各カラムの欠損値の数
    ・各カラムのデータの型
    ・先頭&末尾から5行目まで
    を確認する関数
    Input
    ・df:データフレーム
    '''
    print('行と列の長さ\n{}'.format(df.shape))
    print('-'*50)
    print('各カラムの欠損値の数\n{}'.format(df.isnull().sum()))
    print('-'*50)
    print(df.info())
    print('-'*50)
    print('各カラムのデータ型\n{}'.format(df.dtypes))
    display(df.head(), df.tail())

def calc_pick(df, col):
    '''
    関数内容
    ・データフレームをマージ後、指定カラムに紐づいたレコード数と紐づいたレコード数の割合を算出する関数
    Input
    ・df:データフレーム
    ・col:カラム名
    '''
    print(col + 'カラムに紐づいたレコード数は{}'.format((len(df) - (df[col].isnull().sum()))))
    print(col + 'カラムに紐づいた割合は{:.2%}'.format((len(df) - (df[col].isnull().sum())) / len(df)))


def calc_drop_df_pick(df, drop_df):
    
    print('欠損値がないレコード数は{}'.format((len(drop_df))))
    print('欠損値がないレコード数の割合は{:.2%}'.format((len(drop_df)) / len(df)))

def calc_kaikyu_sizewidth(df, col):
    '''
    関数内容
    ・度数分布表の階級の数と階級の幅を算出する関数
    スタージェスの公式（Sturges' formula）から階級の数を求める。
    → 度数分布やヒストグラム作成時の階級数の目安を得られる公式。nをサンプル数、kを階級数として下記の式で求めることができる。
    → k=1+log2n
　　Input
    ・df:該当データフレーム
    ・col:ヒストグラム描画時にx軸にしたいカラム
    関数使用方法
    ・calc_kaikyu_sizewidth(std_df, 'std_tf_idf')
    '''
    import math
    # 階級の数を算出
    class_size = 1 + math.log(len(df[col]), 2)
    # 階級の幅を算出
    class_width =(max(df[col])-min(df[col])) / class_size#分母は階級の数、分子は範囲

    print('度数分布表の階級数は{}です。'.format(class_size),'階級幅は{}です。'.format(class_width))

def freq_cnt (df):
    '''
    関数内容
    ・['std']カラムの数値の大きさによってグループ分けする関数
    '''
    if  0.1 < df['std'] < 0.3:
        clas = 1
        return clas
    elif 0.3 <= df['std'] < 0.5:
        clas = 2
        return clas
    elif 0.5 <= df['std'] < 0.7:
        clas = 3
        return clas
    elif 0.7 <= df['std'] < 0.9:
        clas = 4
        return clas

def create_frequency_distribution(df, col, rename_col):
    '''
    関数内容
    ・度数分布表を作成する関数
    Input
    ・df:該当データフレーム
    ・col:度数、累積度数、相対度数、累積相対度数を算出したいカラム
    ・rename_col:reset_indexメソッド後にカラム名[index]から変更したい「名称」
    関数使用方法
    ・create_frequency_distribution(df, 'original_type', '単語')
    '''
    df = df[col].value_counts().reset_index().rename(columns={'index':rename_col, col:'度数'})
    df['累積度数'] = df['度数'].cumsum()
    df['相対度数'] = round((df['度数'] / sum(df['度数'])), 3)
    df['累積相対度数'] = round((df['累積度数'] / sum(df['度数'])), 3)

    return df

def create_frequency_distribution_sort_col(df, col, rename_col):
    '''
    関数内容
    ・度数分布表(要素の名前昇順バージョン)を作成する関数
    Input
    ・df:該当データフレーム
    ・col:度数、累積度数、相対度数、累積相対度数を算出したいカラム
    ・rename_col:reset_indexメソッド後にカラム名[index]から変更したい「名称」
    関数使用方法
    ・create_frequency_distribution(df, 'original_type', '単語')
    '''
    df = df[col].value_counts().reset_index().rename(columns={'index':rename_col, col:'度数'}).sort_values(rename_col)
    df['累積度数'] = df['度数'].cumsum()
    df['相対度数'] = round((df['度数'] / sum(df['度数'])), 3)
    df['累積相対度数'] = round((df['累積度数'] / sum(df['度数'])), 3)

    return df

def create_pareto_chart(df, col, fontsize, rotation):
    '''
    関数内容
    ・パレート図を描画する関数
    Input
    ・df:データフレーム
    ・col:x軸に指定したいカラム
    ・fontsize:x軸、y軸の目盛り、ラベルの文字サイズ
    ・rotation:x軸ラベルの文字の回転角度
    関数使用方法
    ・create_pareto_chart(freq_micro_meishi_df, '品詞小分類', 18, 90)
    '''
    #x軸y軸の文字サイズを調整
    plt.rcParams['font.size'] = fontsize

    fig, ax1 = plt.subplots(figsize=(20,12))
    data_num = len(df)

    ax1.bar(range(data_num), df['相対度数'])
    ax1.set_xticks(range(data_num))
    plt.xticks(rotation=rotation)
    ax1.set_xticklabels(df[col].tolist())

    ax1.set_xlabel(col, fontsize=fontsize)
    ax1.set_ylabel('全体に占める割合', fontsize=fontsize)

    ax2 = ax1.twinx()
    ax2.plot(range(data_num), df['累積相対度数'], c="k", marker="o")
    ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

def compare_bar_graph(df, df1, x_col, x1_col, y_col, y1_col, xticks_list, x_label, y_label, legend1, legend2, fontsize=18):
    '''
    関数内容
    ・2つの棒グラフを並べて可視化する関数
    関数使用方法
    ・compare_bar_graph(
    freq_u_buy_df, freq_u_sell_df, '格_b', '格_s', '度数', '度数',[i for i in range(1,11)], '格', '企業数', '企業数(b側)', '企業数(s側)'
    )
    '''
    df = df.astype({x_col:int}).sort_values(x_col)
    df1 = df1.astype({x1_col:int}).sort_values(x1_col)

    x1 = df[x_col]
    y1 = df[y_col]

    x2 = df1[x1_col]
    y2 = df1[y1_col]
    
　　#x軸y軸の文字サイズを調整
    plt.rcParams['font.size'] = fontsize

    plt.bar(x1, y1, color='b', width=-0.4, align='edge', alpha=0.4)
    plt.bar(x2, y2, color='r', width=0.4, align='edge', alpha=0.4)

    plt.xticks(xticks_list, xticks_list)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    labels=[legend1, legend2]
    plt.legend(labels)

def complement_na_from_oneside(df, col1, col2, new_col):
    '''
    関数内容
    ・あるカラムの欠損値を別のカラムの要素で補完する関数
    関数使用方法
    ・merge_df2 = complement_na_from_oneside(merge_df, 'コード_z', 'コード_c', '小分類')
    ・merge_df2 = complement_na_from_oneside(merge_df2, 'G番号', 'G', 'G番号_new').rename(columns={'G番号_new':'G番号'})
    '''
    df[new_col] = np.where(df[col1].notnull(), df[col1], np.where(df[col2].notnull(), df[col2], np.nan))
    # 補完に使ったカラム、元のカラムを削除
    df = df.drop([col1, col2], axis=1)
    return df

def complement_na_from_oneside_3col(df, col1, col2, col3, new_col):
    '''
    関数内容
    ・指定カラムの欠損値を別の指定カラムの要素で補完する関数
    '''
    df[new_col] = np.where(df[col1].notnull(), df[col1], np.where(df[col2].notnull(), df[col2], np.where(df[col3].notnull(), df[col3], np.nan)))
    # 補完に使ったカラム、元のカラムを削除
    df = df.drop([col1, col2, col3], axis=1)
    
    return df

def judge_duplication_of_two_columns(df, col1, col2, col3):
    '''
    関数内容
    ・2つのカラムの組み合わせが重複かどうかを判定する関数
    Input
    ・df:データフレーム
    ・co1, col2:重複判定したい2つのカラム
    ・col3:カウントしたいカラム(重複判定したいだけなのでdfに格納されているどのカラムでも問題ない)
    '''
    if len(df) == len(df.groupby([col1, col2])[[col3]].count()):
        print(col1 + 'と' + col2 +'の組み合わせに重複はありません。')
    else:
        print(col1 + 'と' + col2 +'の組み合わせに重複があります。')

def compare_bar_graph(df, df1, x_col, x1_col, y_col, y1_col, xticks_list, x_label, y_label, legend1, legend2, int_flag, fontsize=18, rotation=90):
    '''
    関数内容
    ・2つの棒グラフを並べて可視化する関数
    関数使用方法
    ・compare_bar_graph(
    f_macro_buy_df, f_macro_sell_df, '分類名_b', '分類名_s', '度数', '度数', None,
    '分類名', '企業数', '企業数(買側)', '企業数(被買側)', False
    )
    '''
    if int_flag is True:
        df = df.astype({x_col:int}).sort_values(x_col)
        df1 = df1.astype({x1_col:int}).sort_values(x1_col)
    else:
        pass

    x1 = df[x_col]
    y1 = df[y_col]

    x2 = df1[x1_col]
    y2 = df1[y1_col]

    #x軸y軸の文字サイズを調整
    plt.rcParams['font.size'] = fontsize

    plt.bar(x1, y1, color='b', width=-0.4, align='edge', alpha=0.4)
    plt.bar(x2, y2, color='r', width=0.4, align='edge', alpha=0.4)

    plt.xticks(xticks_list, xticks_list)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation)

    labels=[legend1, legend2]
    plt.legend(labels)

def create_frequency_ratio_graph(df, set_index_col, ratio_col1, ratio_col2, sort_col, y_label, tate_yoko_flag, tate_size, yoko_size, int_flag, fontsize=18):
    '''
    関数内容
    ・2つの度数の割合を縦&横棒グラフで可視化する関数
    関数使用方法
    ・縦棒グラフ選択：create_frequency_ratio_graph(f_merge_df, '大分類名', 'buy_ratio', 'sell_ratio', None, '企業数の割合', 'tate', 15, 10, False)
    ・横棒グラフ選択：create_frequency_ratio_graph(f_merge_df2, '中分類名', 'buy_ratio', 'sell_ratio', '度数_buy', '企業数の割合', 'yoko', 15, 30, False)
    '''
    font_size = fontsize
    graph_params = {
        'axes.labelsize':font_size,
        'axes.titlesize':font_size + 2,
        'xtick.labelsize':font_size,
        'figure.figsize' : [tate_size, yoko_size],
        'ytick.labelsize' : font_size,
        'legend.fontsize' : font_size,
        'font.family' : 'IPAexGothic'
    }
    plt.rcParams.update(**graph_params)

    # カラムがint型(順序尺度)かを判定する条件分岐 順序尺度の場合は予め昇順にしておく
    if int_flag is True:
        df = df.astype({set_index_col:int}).sort_values(set_index_col)
    else:
        pass

    if tate_yoko_flag == 'tate':
        ratio_tate_graph_df = df.set_index(set_index_col)
        ratio_tate_graph_df[[ratio_col1, ratio_col2]].plot.bar(stacked=True)
        plt.ylabel(y_label)

    elif tate_yoko_flag == 'yoko':
        ratio_yoko_graph_df = df.set_index(set_index_col).sort_values(sort_col)
        ratio_yoko_graph_df[[ratio_col1, ratio_col2]].plot.barh(stacked=True)
        plt.xlabel(y_label)

    else:
        print("tate_yoko_flagには'tate'か'yoko'という文字列を与えてください")

def check_string_length(df, col, threshold, num_of_display):
    '''
    関数内容
    ・指定カラム(object型)の各要素の文字列の長さを閾値として、閾値未満のレコードのみ抽出する関数(同時に行列数、先頭から指定行数のレコードも表示する)
    Input
    ・df:データフレーム(予め文字列の長さを確認したいカラムの欠損値を削除している状態が望ましい)
    ・col:各要素の文字列の長さを算出したいカラム
    ・threshold:文字列の長さの閾値(この数値未満のレコードを抽出する)
    ・num_of_display:先頭から表示したいレコード数
    関数使用方法
    ・check_string_length(df, 'コード', 5, 10)
    '''
    df['文字列の長さ'] = df[col].map(lambda x: len(str(x)))
    
    under_threshold_df = df[df['文字列の長さ'] < threshold]

    display(under_threshold_df.shape, under_threshold_df.head(num_of_display))

    return under_threshold_df

def change_micro_code_from_string_length(df, threshold=5):
    '''
    関数内容
    ・文字列の長さが閾値未満(デフォルト5文字未満)ならば[業種]カラムの要素を[コード]カラムの要素に変える関数
    Input
    ・df:データフレーム
    '''
    if df['文字列の長さ'] < threshold:
        df['業種'] = df['コード']
    else:
        pass

    return df

def get_normalized_text(text):
    '''
    関数内容
    ・テキストの文字列の正規化を行う関数
    Input
    ・text：テキスト文(str型)
    関数使用方法
    ・df['名称'] = df['名称'].progress_map(get_normalized_text)
    '''
    import neologdn
    #np.nanがstr型ではなくfloat型とされるためstr型以外はneologdn.normalizeメソッドを実施しないように条件分岐
    if not isinstance(text, str):
        pass
    else:
        text_normalization = neologdn.normalize(text)
        # 大文字英字を小文字英字に変更
        text_normalization = text_normalization.lower()

        return text_normalization

def extract_contains_string_df(df, col, string):
    '''
    関数内容
    ・指定カラムに特定文字が含まれているレコードのみ抽出する関数
    関数使用方法
    ・extract_contains_string_df(all_df2, '名前', '口')
    　del_list = ['口', '工', '幹', '破産']
      kouza_df, kou_df, kan_df, hasan_df = [extract_contains_string_df(all_df2, '名前', string) for string in del_list]
    '''
    out_df = df[df[col].str.contains(string, na=False)]

    return out_df

def diff_df_length(before_df, after_df):
    '''
    関数内容
    ・削除したレコード数を確認する関数
    '''
    display('削除したレコード数は{}レコードです。'.format(len(before_df) - len(after_df)))

def save_excel(df, string, col='取引先名漢字'):
    '''
    関数内容
    ・指定カラムに特定文字が含まれているレコードのみ抽出してxlsxファイルで出力する関数
    Input
      string:特定文字列 or 特定正規表現
    '''
    out_df = df[df[col].str.contains(string, na=False)]

    return out_df[[col]].to_excel(col + 'カラムに「' + string +'」 が含まれるレコード.xlsx', index=False)

def judge_include_value(df, value, base_col):
    '''
    関数内容
    ・指定したカラムの値が別のカラムの値の中に含まれているかどうかを判定する関数
    Input
    ・df :該当データフレーム
    ・value :含まれているかどうか判定したい値(str)
    ・base_col : カラム名(このカラムの値の中にvalueが含まれているかどうかを判定)(str)
    Returns
    ・type : 要素が含まれていれば「1」、含まれていなければ「0」を返す(str)
    関数使用方法
    ・df['flag'] = [judge_include_number(df, value, 'id_b') for value in df['id'].tolist()]
    '''
    base_list = df[base_col].tolist()

    if value in base_list:
        return '1'
    else:
        return '0'

def save_excel(df, string):
    '''
    関数内容
    ・[取引先]カラムに特定文字が含まれているレコードのみ抽出しxlsxファイルで出力する関数
    '''
    out_df = df[df['取引先'].str.contains(string, na=False)]

    return out_df[['取引先']].to_excel('取引先カラムに「' + string +'」 が含まれるレコード.xlsx', index=False)

def save_col_name_df(df, name):
    '''
    関数の内容
     ・データフレームのカラム名のみ抽出したデータフレームを作成し、.xlsxファイルで出力する関数
    Input
     ・df：データフレーム
     ・name：「○○のカラム名一覧」の○○部分に加えたい文字列
     Return
      ・データフレーム先頭から5行、カラム名のみ抽出したデータフレームを.xlsxファイルで出力

    関数使用方法
     ・save_col_name_df(master_df, 'マスタデータ')

    '''

    df_col_list = df.columns.tolist()

    col_df = pd.DataFrame({
        name + 'のカラム一覧':df_col_list
    })


    return col_df.head(), col_df.to_excel(name + 'のカラム一覧.xlsx', index=False)

def exclude_outliers(df, col):
    '''
    関数内容
    ・外れ値をnanに変換して削除する関数
    '''

    # 第一四分位数、第三四分位数
    q1 = df[col].describe()['25%']
    q3 = df[col].describe()['75%']

    # 四分位範囲
    iqr = q3 - q1

    lower_threshold = q1 - (1.5 * iqr)
    upper_threshold = q3 + (1.5 * iqr)

    df[col] = df[col].map(lambda x: np.nan if x < lower_threshold or x > upper_threshold else x)
    df = df.dropna(subset=[col])

    return df


def flag_outliers(df, col):
    '''
    関数内容
    ・外れ値にフラグをつけ削除する関数
    '''

    # 第一四分位数、第三四分位数
    q1 = df[col].describe()['25%']
    q3 = df[col].describe()['75%']

    # 四分位範囲
    iqr = q3 - q1

    lower_threshold = q1 - (1.5 * iqr)
    upper_threshold = q3 + (1.5 * iqr)

    df['outliers_flag'] = df[col].map(lambda x: -1 if x < lower_threshold or x > upper_threshold else 0)
    # 外れ値以外のレコードのみ抽出
    df = df[df['outliers_flag'] == 0]

    return df

def add_value_flag(df, name, raw_df, key_col):
    '''
    関数内容
      ・元のデータフレームと紐づけを試みて元のデータフレームに紐づいたかどうかを判定するフラグを付与したデータフレームを作成する関数
      (紐づけようとしているデータフレームの既存のカラムはkey_colと関数内で指定するnew_colのみ残すようにしている)
    Input
      ・df：紐づけようとしているデータフレーム
      ・name：フラグを付与した新規カラム['is_' + name + '_flag']の[name]部分に入れたい名前
      ・raw_df：紐づけられる元のデータフレーム
      ・key_col：raw_dfとdfをマージする際にキーにするカラム
    Return
      ・元のデータフレームに紐づいたかどうかを判定するフラグを付与したデータフレーム

    使い方
    　tmp_merge_df = add_value_flag(tmp_df, 'dummy', tmp_raw_df, 'id')
    '''

    new_col = 'is_' + name + '_flag'

    df[new_col] = '1'
    df = df[[key_col] + [new_col]]

    merge_df = pd.merge(raw_df, df, on=key_col, how='left', suffixes=['_r', '_d'])
    merge_df[new_col] = merge_df[new_col].fillna('0')

    return merge_df

def add_value_flag_to_df(df, name, raw_df, key_col):
    '''
    関数内容
      ・元のデータフレームと紐づけを試みて元のデータフレームに紐づいたかどうかを判定するフラグを付与したデータフレームを作成する関数
      (紐づけようとしているデータフレームの既存のカラムすべてを最後まで残している)
    Input
      ・df：紐づけようとしているデータフレーム
      ・name：フラグを付与した新規カラム['is_' + name + '_flag']の[name]部分に入れたい名前
      ・raw_df：紐づけられる元のデータフレーム
      ・key_col：raw_dfとdfをマージする際にキーにするカラム
    Return
      ・元のデータフレームに紐づいたかどうかを判定するフラグを付与したデータフレーム

    使い方
    　tmp_merge_df = add_value_flag(tmp_df, 'dummy', tmp_raw_df, 'CIF')
    '''
    new_col = 'is_' + name + '_flag'

    df[new_col] = '1'

    merge_df = pd.merge(raw_df, df, on=key_col, how='left', suffixes=['_r', '_d'])
    merge_df[new_col] = merge_df[new_col].fillna('0')

    return merge_df

def add_judge_col(df, col1, col2, new_col_name):
    '''
    関数内容
     ・指定カラムの値がどちらも「0」であるなら0を、そうでなければ1を付与した新規カラムを作成する関数

    Input
     ・df：データフレーム
     ・col1：「0」or「1」2つの値が含まれるcol2とは別のカラム
     ・col2：「0」or「1」2つの値が含まれるcol1とは別のカラム
     ・new_col_name：新規カラム名

    Return
     ・指定カラムの値がどちらも「0」であるなら0を、そうでなければ1を付与した新規カラムが追加されたデータフレーム

    関数使用方法
     ・df1 = add_judge_col(df, 'b', 's', 'is_0_flag')
    '''

    df[new_col_name] = np.where(((df[col1] == 0) & (df[col2] == 0)), 0, 1)
    # numpyでは文字列型を使うことができないのでここで文字列型に変換
    df[new_col_name] = df[new_col_name].astype(str)

    return df

def add_judge_str_col(df, col1, col2, new_col_name):
    '''
    関数内容
     ・指定カラムの値がどちらも「'0'」(※この0は文字列型)であるなら0を、そうでなければ1を付与した新規カラムを作成する関数
    Input
     ・df：データフレーム
     ・col1：「'0'」or「'1'」2つの値が含まれるcol2とは別のカラム
     ・col2：「'0'」or「'1'」2つの値が含まれるcol1とは別のカラム
     ・new_col_name：新規カラム名

    Return
     ・指定カラムの値がどちらも「'0'」であるなら0を、そうでなければ1を付与した新規カラムが追加されたデータフレーム

    関数使用方法
     ・df1 = add_judge_str_col(df, 'b', 's', 'is_0_flag')
    '''
    df[new_col_name] = np.where(((df[col1] == '0') & (df[col2] == '0')), 0, 1)
    # numpyでは文字列型を使うことができないのでここで文字列型に変換
    df[new_col_name] = df[new_col_name].astype(str)

    return df

def add_judge_str_3col(df, col1, col2, col3, new_col_name):
    '''
    関数内容
     ・指定カラムの値がどちらも「'0'」(※この0は文字列型)であるなら0を、そうでなければ1を付与した新規カラムを作成する関数

    Input
     ・df：データフレーム
     ・col1：「'0'」or「'1'」2つの値が含まれるcol2、col3とは別のカラム
     ・col2：「'0'」or「'1'」2つの値が含まれるcol1、col3とは別のカラム
     ・col3：「'0'」or「'1'」2つの値が含まれるcol1、col2とは別のカラム
     ・new_col_name：新規カラム名

    Return
     ・指定カラムの値がどちらも「'0'」であるなら0を、そうでなければ1を付与した新規カラムが追加されたデータフレーム

    関数使用方法
     ・df1 = add_judge_str_3col(df, 'b', 's', 'c', is_0_flag')
    '''

    df[new_col_name] = np.where(((df[col1] == '0') & (df[col2] == '0')  & (df[col3] == '0')), 0, 1)
    # numpyでは文字列型を使うことができないのでここで文字列型に変換
    df[new_col_name] = df[new_col_name].astype(str)

    return df

def create_compare_values_flag(col1, col2):
    '''
    関数内容
    ・2つのカラムの値の大小を比較して0 or 1 フラグを立てる関数
    Input
    ・col1：col2に指定した以外のカラム
    ・col2：col1に指定した以外のカラム
    関数使用方法
    ・cross_df['比較_flag'] = cross_df.apply(lambda x: create_compare_values_flag(x['buy'], x['sell']), axis=1)
    '''
    if col1 < col2:
        flag = 0
    elif col1 >= col2:
        flag = 1

    return flag

def design_range(col, new_col, num, df=cross_df):
    '''
    関数内容
    ・該当カラムの値の範囲が広すぎる場合に、Kmeansを使うことでレンジを狭める関数
    Input
    ・col：クラスタ化したいカラム
    ・new_col：予測したクラスタを格納し、指定データフレームに格納する新規カラム
    ・num：クラスタの数
    ・df：データフレーム
    関数使用方法
    ・cross_df = design_range('倍率', 'a_s_mag_cluster', 100)
    ※なお、この後にはクラスタの番号が倍率の大小を表していないのでクラスタに分ける前の値を昇順にしてからクラスタの名称を変更することが多い
    g_as_df = cross_df.groupby('a_s_mag_cluster')[['倍率']].mean().sort_values('倍率').reset_index()
    g_as_df['資産/売上高_倍率クラスター'] = range(0, len(g_as_df))
    '''

    from sklearn.cluster import KMeans

    # reshape(1, -1)を使わないとExpected 2D array, got 1D array insteadとエラーが発生する
    # 参考：http://noralog.com/data-analytics-1
    mag_array = np.array(df[col].tolist()).reshape(1, -1)
    mag_array_T = mag_array.T

    pred = KMeans(n_clusters=num).fit_predict(mag_array_T)

    df[new_col] = pred

    return df

def combination_columns_name(col_list1, col_list2):
    '''
    関数内容
    ・2つのカラムの値の大小を比較して0 or 1 フラグを立てる関数
    Input
    ・col_list1：カラムが格納されたリスト
    ・col_list2：col_list1に格納されている各カラム名の末尾に追加したい文字列
    関数使用方法
    ・tmp_col_list = combination_columns_name(tmp_necessary_col_list, tmp_end_name_col_list)
    '''
    import itertools
    combi_col_list = [col1 + col2 for col1, col2 in itertools.product(col_list1, col_list2)]

    return combi_col_list

def add_complement_df(df1, code_buy_col, code_sell_col):
    '''
    関数内容
    ・買収側業種名と売却側業種名の要素が共通になるよう補完し、その補完分を元データフレームに追加する関数
    '''  
    '''買収側業種コードのユニークな要素をリストで抽出'''
    buy_list = df1[code_buy_col].drop_duplicates().tolist()
    '''売却側業種コードのユニークな要素をリストで抽出'''
    sell_list = df1[code_sell_col].drop_duplicates().tolist()
    '''買収側業種コード、売却側業種コードを比較し、どちらか一方にしかない値を抽出'''
    diff_set = set(buy_list)^set(sell_list)#2つのリストを比較し、重複していない要素のみ抽出
    diff_list = list(diff_set)#  set型をlist型に変換
    '''抽出した重複していない要素のうち、買収側業種コードに含まれていない要素のみ抽出'''
    diff_buy_list= list(set(diff_list) - set(buy_list))
    '''抽出した重複していない要素のうち、売却側業種コードに含まれていない要素のみ抽出'''
    diff_sell_list= list(set(diff_list) - set(sell_list))
    '''買収側業種コード、売却側業種コードに含まれていない要素の組み合わせの総当たりを抽出'''
    import itertools
    buy_sell_list = [[x, y] for x, y in itertools.product(diff_buy_list, diff_sell_list)]

    '''組み合わせの総当たり分と各カラム=0(or 0.0)を格納したデータフレームを作成'''
    add_df = pd.DataFrame(buy_sell_list,columns=[code_buy_col, code_sell_col])
    add_df['件数'] = 0
    add_df['成約数'] = 0
    add_df['倍率'] = 0.0
    add_df['成約率'] = 0.0
    add_df['倍率'] = add_df['倍率'].astype(float)
    add_df['成約率'] = add_df['成約率'].astype(float)

    '''新規作成したデータフレームを元のデータフレームに縦に連結'''
    df2 = pd.concat([df1, add_df]).reset_index(drop=True)#インデックス番号を振り直すのみ、列側には移動させない

    return df2

def calc_groupby_median(df, col1, col2, main_col):
    '''
    関数内容
    ・2つのカラム(col1,col2)の組み合わせ毎に件数、成約数、成約率、main_colに指定したカラムの中央値を算出する関数
    '''
    df = df[['success', col1, col2, main_col]]
    df['件数'] = df['success']
    df['成約数'] = df['success']

    df = df.groupby([col1, col2]).agg({'件数':'count', '成約数':'sum', main_col:'median'})

    df['成約率'] = df['成約数'] / df['件数']

    # 件数を降順にする
    df = df.reset_index().sort_values('件数', ascending=False)

    return df

def create_mean_pivot_table(df, index, columns, values):
    '''
    関数内容
    ・ピボットテーブルを作成する関数
    '''
    pivot_df = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc='mean',
        fill_value=0
    )

    return pivot_df

def create_add_num_heatmap(pivot_df, context):
    '''
    関数内容
    ・ヒートマップを描画する関数
    '''
    sns.set_context(context)
    sns.heatmap(pivot_df.sort_index(ascending=False), annot=True, fmt='g', square=True, cmap = 'coolwarm')

# 倍率を算出する関数1
def calc_magnification(numerator_col, denominator_col):
    '''
    関数内容
    ・分母の値が「0」もしくは分子、分母のどちらかが欠損値の場合は倍率を「-1」、それ以外は倍率を算出する関数
    Input
    ・numerator_col:倍率算出時に分子にしたいカラム
    ・denominator_col:倍率算出時に分母にしたいカラム
    関数使用方法
    ・df['売上高/売上高_倍率'] = df.apply(lambda x: calc_magnification(x['売上高_buy'], x['売上高_sell']), axis=1)
    '''
    ratio = -1

    if (str(numerator_col).replace('.', '').isdecimal()) and (str(denominator_col).replace('.', '').isdecimal()):

        numerator_col = float(numerator_col)
        denominator_col = float(denominator_col)

        if denominator_col != 0:
            ratio = numerator_col / denominator_col

    ratio = float(ratio)

    return ratio

# 倍率を算出する関数2
def calc_magnification(numerator_col, denominator_col):
    '''
    関数内容
    ・分母の値が「0」もしくは分子、分母のどちらかが欠損値の場合は倍率を「-1」、それ以外は倍率を算出する関数
    Input
    ・numerator_col:倍率算出時に分子にしたいカラム
    ・denominator_col:倍率算出時に分母にしたいカラム
    関数使用方法
    ・df['売上高/売上高_倍率'] = df.apply(lambda x: calc_magnification(x['売上高_buy'], x['売上高_sell']), axis=1)
    '''
    if denominator_col == 0:
        ratio = -1
    elif np.isnan(numerator_col)  or np.isnan(denominator_col):
        ratio = -1
    else:
        ratio = numerator_col / denominator_col

    return ratio

def text_to_dataframe(text, separation, col_names:list):
    '''
    関数内容
    ・txtファイルをデータフレームとして読み込む関数
    Input
    ・text：テキストファイル名
    ・separation：タブ区切り('\t')orカンマ区切り(',')を指定
    ・col_names：指定したいカラム名、リスト型で与える必要あり
    関数使用方法
    ・test_text = text_to_dataframe('neko.txt.ginza', '\t', ['index', 'surface', 'original', 'type'])
    '''
    text_df = pd.read_csv(text, sep=separation, names=col_names)

    return text_df
