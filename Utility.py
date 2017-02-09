u""" あるディレクトリにおける、指定listされた拡張子のファイルのみをlist化して返す"""
def check_file(path , ext):
    #http://qiita.com/icecream177/items/3d8872f024da2e8feca0
    import os
    import os.path

    _ch = os.listdir(path)
    ch_e = []

    for file_name in _ch:
        _root, _ext = os.path.splitext(file_name)

        if _ext == ext:
            ch_e.append(file_name)
        else:
            pass
    return ch_e