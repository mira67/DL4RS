"""
utils
Author: Qi Liu
"""

import ConfigParser

def cfgmap(cfg,section):
    dict1 = {}
    options = cfg.options(section)
    for option in options:
        try:
            dict1[option] = cfg.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

def read_config():
    config = ConfigParser.ConfigParser()
    config.read("config.ini")
    #datapath
    p = {}
    p['train_path'] = cfgmap(config,"Workspace")['train_path']
    p['result_path'] = cfgmap(config,"Workspace")['result_path']
    p['test_path'] = cfgmap(config,"Workspace")['test_path']
    p['model_path'] = cfgmap(config,"Workspace")['model_path']
    p['test_result_csv'] = cfgmap(config,"Workspace")['test_result_csv']
    #parameters
    p['fea_num'] = config.getint('Settings','fea_num')
    p['out_num'] = config.getint('Settings','out_num')
    p['bsize'] = config.getint('Settings','batch_size')
    p['iters'] = config.getint('Settings','nb_epoch')
    #graphics
    p['plot_on'] = config.getint('Settings','plot_on')
    # model
    p['model_id'] = config.getint('Model','model_id')
    p['model_name'] = cfgmap(config,"Model")['model_name']
    p['model_name2'] = cfgmap(config,"Model")['model_name2']
    p['verbose_on'] = config.getboolean('Model','verbose_on')
    p['kfold'] = config.getint('Model','kfold')
    p['n_splits'] = config.getint('Model','n_splits')

    # sql
    p['train_sql'] = cfgmap(config,"SQL")['train_sql']
    p['test_sql'] = cfgmap(config,"SQL")['test_sql']
    p['csvtosql'] = cfgmap(config,"SQL")['csvtosql']
    return p
