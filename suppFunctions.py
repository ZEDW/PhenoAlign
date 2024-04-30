# -*- coding: utf-8 -*-
# a series of supporting functions for main functions 

import re
import os
import json
import time
import datetime
import pandas as pd
import pymysql
import pickle
import hashlib
import difflib
import requests
import orjson
import shutil
import random
from itertools import groupby
from collections import Counter
# from pymysql import escape_string
from bs4 import BeautifulSoup
from markdown import markdown
from operator import itemgetter
from dateutil.relativedelta import relativedelta
from flashtext4en import KeywordProcessor4en
from flashtext4cn import KeywordProcessor4cn
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

BERT_PATH = '../SimClassifier'
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH, num_labels=3)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to('cpu')

######################
# supporting functions
######################

# demo functions
def hello_world():
    return "hello world"


######################
# 获取SSU相关表格的数据
######################
# SSU_TYPE 包含的核心词语义标签
def get_ssu_type_of_tags( db_info_dict ):
    # T019 属于表型核心词
    ssu_type_of_tags = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # 读取SSU关联属性表数据
    # build sql command
    sql_cmd = "select SSU_TYPE, TUI, STY from tags_included_in_ssu"
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()  

    #
    for row in rows:
        # key -- TUI
        # value -- SSU_TYPE
        # store
        ssu_type_of_tags.setdefault( row[1], row[0] )

    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   

    # return 
    return ssu_type_of_tags


# SSU关联属性表
def get_info_of_ssu_attributes( db_info_dict ):
    # 存在情况是表型核心词的属性, 其英文名称是Present
    info_of_ssu_attributes = {}


    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # 读取SSU关联属性表数据
    # build sql command
    sql_cmd = "select ATTR_CN_NAME, SSU_TYPE, ATTR_EN_NAME, ATTR_ORDER, VALUE_SET_TYPE, ALLOW_MULTI_VALUE from def_ssu_attributes"
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()  

    #
    for row in rows:
        # key -- attr_cn_name
        attr_cn_name = row[0]

        # value
        attr_info_dict = {}
        attr_info_dict.setdefault( "SSU_TYPE", row[1] )
        attr_info_dict.setdefault( "ATTR_EN_NAME", row[2] )
        attr_info_dict.setdefault( "ATTR_ORDER", row[3] )
        attr_info_dict.setdefault( "VALUE_SET_TYPE", row[4] )
        attr_info_dict.setdefault( "ALLOW_MULTI_VALUE", row[5] )

        # store
        info_of_ssu_attributes.setdefault( attr_cn_name, attr_info_dict)

    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   

    # return 
    return info_of_ssu_attributes


# 属性关联取值表
# 加入 excluded_attributes, 不考虑这一属性, 默认为空列表
# 加入 included_attributes, 仅考虑这一属性，默认为空列表
def get_info_of_std_values( db_info_dict, excluded_attributes, included_attributes  ):
    # 标准取值的信息和触发词信息
    info_of_std_attributes = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # 读取SSU属性标准取值表数据
    # build sql command
    sql_cmd = "select VALUE_CN_NAME, VALUE_EN_NAME, SSU_TYPE, ATTR_CN_NAME, VALUE_ORDER, VALUE_TRIGGER_CN, VALUE_TRIGGER_EN " \
                "from def_ssu_std_values"
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()  

    #
    for row in rows:
        # key -- value_cn_name
        value_cn_name = row[0]

        # 该行数据对应的属性名称
        attr_cn_name = row[3]

        # 如果 excluded_attributes 不为空
        # 如果该行数据对应的 attr_cn_name 不在 excluded_attributes 列表中
        # 跳过
        if type( excluded_attributes ) == list :
            if len( excluded_attributes ) != 0:
                if attr_cn_name in excluded_attributes:
                    continue

        # 如果 included_attributes 不为空
        # 如果该行数据对应的 attr_cn_name 不在 included_attributes 列表中
        # 跳过
        if type( included_attributes ) == list :
            if len( included_attributes ) != 0:
                if attr_cn_name not in included_attributes:
                    continue


        # value
        value_info_dict = {}
        value_info_dict.setdefault( "VALUE_EN_NAME", row[1] )
        value_info_dict.setdefault( "SSU_TYPE", row[2] )
        value_info_dict.setdefault( "ATTR_CN_NAME", row[3] )
        value_info_dict.setdefault( "VALUE_ORDER", row[4] )
        value_info_dict.setdefault( "VALUE_TRIGGER_CN", row[5] )
        value_info_dict.setdefault( "VALUE_TRIGGER_EN", row[6] )

        # store
        info_of_std_attributes.setdefault( value_cn_name, value_info_dict)

    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   


    # return 
    return info_of_std_attributes


# 读取段落标题核心词触发模式表
# 生成用于flashtext的字典形式
def get_section_triggers_info( db_info_dict ):
    # 
    dict_of_section_triggers = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # build sql command
    sql_cmd = "select section_prefered_name, section_trigger_pattern from section_triggers_info"
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()  

    #
    for row in rows:
        # std_name, trigger_pattern
        section_std_name       = row[0]
        section_trigger_pattern = row[1]

        # 
        section_clean_name = '标题||' + section_std_name 

        #
        if section_clean_name not in dict_of_section_triggers:
            dict_of_section_triggers.setdefault( section_clean_name, [] )

        #
        for section_trigger_word in section_trigger_pattern.split('|'):
            if section_trigger_word not in dict_of_section_triggers[section_clean_name]:
                dict_of_section_triggers[section_clean_name].append( section_trigger_word )


    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   

    # return 
    return dict_of_section_triggers


#
def build_dict_of_std_values(info_of_std_attributes, lang_code):
    # clean_name: [list of unclean_names]
    dict_of_std_values = {}

    # clean_name 
    # 什么语义结构单元-什么属性-的标准取值
    # 表型核心词||严重程度||标准取值

    for value_cn_name in info_of_std_attributes:
        # 
        value_info_dict = info_of_std_attributes[value_cn_name]

        # clean_name 
        # 触发词(严重) 对应的标准词是轻度,  是一种严重程度, 修饰的是 表型核心词
        clean_name = "||".join(  [value_cn_name, value_info_dict["ATTR_CN_NAME"], value_info_dict["SSU_TYPE"]] ) 

        if clean_name not in dict_of_std_values:
            # [轻度]
            dict_of_std_values.setdefault( clean_name, [value_cn_name] )

        # unclean_name 
        value_trigger_words = []

        if lang_code == 'cn':
            if value_info_dict["VALUE_TRIGGER_CN"] not in [None, ""]:
                value_trigger_words = value_info_dict["VALUE_TRIGGER_CN"].split("||")
        elif lang_code == 'en':
            if value_info_dict["VALUE_TRIGGER_EN"] not in [None, ""]:
                value_trigger_words = value_info_dict["VALUE_TRIGGER_EN"].split("||")            

        #
        for value_trigger_word in value_trigger_words:
            if value_trigger_word not in dict_of_std_values[clean_name]:
                dict_of_std_values[clean_name].append( value_trigger_word )

    return dict_of_std_values


# 正常范围取值表
def get_info_of_ref_ranges( db_info_dict  ):
    # 求解目标
    info_of_ref_ranges = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # build sql command
    sql_cmd = "select * from observable_ent_ranges" 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()  

    #
    for row in rows:
        # 可观测对象
        ent_first_name = row[1]

        # 可观测对象来源 (Default: NULL)
        sample_name = row[3]

        if sample_name in [None, ""]:
            sample_name = ""

        # 正常范围
        lower_limit = row[5]
        upper_limit = row[6]

        # 测量单位进行小写处理
        str_of_unit = row[7]

        if str_of_unit in [None, ""]:
            str_of_unit = ""    
        else:    
            str_of_unit = str_of_unit.upper()


        # 适用人群
        gender_code = row[9]
        lower_age   = row[10]
        upper_age   = row[11]
        unit_of_age = row[12]

        if gender_code in [None, ""]:
            gender_code = ""

        if lower_age in [None, ""]:
            lower_age = ""

        if upper_age in [None, ""]:
            upper_age = ""

        if unit_of_age in [None, ""]:
            unit_of_age = ""                                


        # key 
        ent_id = ent_first_name + '@' + sample_name

        # ent_id: { unit1:[(range1), (range2)], unit2  }
        # 这种数据结构便于提取 ent_id 的 units_in_kb 
        if ent_id not in info_of_ref_ranges:
            info_of_ref_ranges.setdefault( ent_id, {} )

        # 
        if str_of_unit not in info_of_ref_ranges[ent_id]:
            info_of_ref_ranges[ent_id].setdefault( str_of_unit, [] )

        # store the range as a dict
        tmpinfo = {}
        tmpinfo.setdefault( "lower_limit", lower_limit )
        tmpinfo.setdefault( "upper_limit", upper_limit )
        tmpinfo.setdefault( "str_of_unit", str_of_unit )
        tmpinfo.setdefault( "gender_code", gender_code )
        tmpinfo.setdefault( "lower_age",   lower_age)
        tmpinfo.setdefault( "upper_age",   upper_age)
        tmpinfo.setdefault( "unit_of_age", unit_of_age )

        # 
        info_of_ref_ranges[ent_id][str_of_unit].append( tmpinfo )

    #
    return info_of_ref_ranges


# 可观测实体名称表 observable_ent_names
def get_corewords_from_LatteKB(db_info_dict, ssu_type_of_tags):
    # 求解目标
    # clean_name: [unclean_name]
    corewords_from_LatteKB = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # build sql command
    sql_cmd = "select ent_all_names, ent_first_name, ent_cui, ent_tui, ent_sty from observable_ent_names where mask_tag = 'N' " 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()      


    #
    for row in rows:        
        # ACP染色|ACP
        ent_all_names = row[0].split('|')
        ent_first_name = row[1]
        ent_cui = row[2]

        #        
        ent_tui = row[3]
        ent_sty = row[4]

        # 核心词类型 
        # 设置为 测量核心词
        core_type = '测量核心词'

        # 在此处没什么必要，因为都是测量核心词
        # if ent_tui in ssu_type_of_tags:
        #     core_type = ssu_type_of_tags[ent_tui]        


        # entity information as clean_name (对照 MRCONSO4CN 的 keywords_dict 进行处理)
        # ent_info = term_cui + '||' + term_tui + '||' + term_tag + '||' + core_type
        #         ent_first_name 
        ent_info = ent_first_name + '||' + ent_tui + '||' + ent_sty + '||' + core_type

        # 
        if ent_info not in corewords_from_LatteKB:
            corewords_from_LatteKB.setdefault( ent_info, [] )

        #
        if len( ent_first_name ) >= 2:
            if ent_first_name not in corewords_from_LatteKB[ent_info]:
                corewords_from_LatteKB[ent_info].append( ent_first_name )

        # 
        for ent_alias_name in ent_all_names:
            if len( ent_alias_name ) >=2 :
                if ent_alias_name not in corewords_from_LatteKB[ent_info]:
                    corewords_from_LatteKB[ent_info].append( ent_alias_name )

    #
    return corewords_from_LatteKB


# 基于LATTE知识库，将 可观测实体名称 映射到 ent_first_name
def get_mapping_of_observable_ent_names( corewords_from_LatteKB ):
    #
    mapping_of_observable_ent_names = {}

    # corewords_from_LatteKB.setdefault( ent_info, [] )
    for ent_info in corewords_from_LatteKB:
        #
        ent_first_name = ent_info.split("||")[0]

        #
        mapping_of_observable_ent_names.setdefault( ent_first_name, ent_first_name )


        # 
        ent_all_names = corewords_from_LatteKB[ent_info]

        for ent_alias_name in ent_all_names:
            mapping_of_observable_ent_names.setdefault( ent_alias_name, ent_first_name )

    #
    return mapping_of_observable_ent_names


# 来自PubTerms的 核心词实体名称表 pub_terms
# pub_terms 中的术语名称的标准化
# 优先考虑 CUI 化处理，若不具有确定的CUI
# 则用 pub_syns 中的知识进行标准化处理
def get_corewords_from_PubTerms(db_info_dict, ssu_type_of_tags, prefered_name_from_PubSyns):
    # 求解目标
    # clean_name: [unclean_name]
    corewords_from_PubTerms = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # build sql command
    sql_cmd = "select term_cn, term_cui, fully_confirmed, term_tui, term_sty from pub_terms where mask_tag = 'N' " 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()      

    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   

    #
    for row in rows:        
        # 
        term_cn  = row[0]
        term_cui = row[1]
        fully_confirmed = row[2]

        #        
        ent_tui = row[3]
        ent_sty = row[4]

        # 核心词类型 
        # 默认 一般核心词
        core_type = '一般核心词'

        # 根据实际的 TUI 进行更改
        if ent_tui in ssu_type_of_tags:
            core_type = ssu_type_of_tags[ent_tui]


        # entity information as clean_name (对照 MRCONSO4CN 的 keywords_dict 进行处理)
        # ent_info = term_cui + '||' + term_tui + '||' + term_tag + '||' + core_type
        ent_info = ""

        # 如果 CUI 是确定正确的，使用 term_cui 作为标准名称
        if term_cui != 'NULL' and fully_confirmed == 'Y':
            ent_info = term_cui + '||' + ent_tui + '||' + ent_sty + '||' + core_type
        # else 使用 term_cn 占据 ent_cui 的位置
        # [调整] 尝试适用 pub_syns 中的知识进行标准化
        else:
            # 
            std_term_str = term_cn

            if std_term_str in prefered_name_from_PubSyns:
                std_term_str = prefered_name_from_PubSyns[std_term_str]
            #
            ent_info = std_term_str + '||' + ent_tui + '||' + ent_sty + '||' + core_type

        # 
        if ent_info not in corewords_from_PubTerms:
            corewords_from_PubTerms.setdefault( ent_info, [] )

        # 只考虑两个以上字符
        if len(term_cn) >= 2:
            if term_cn not in corewords_from_PubTerms[ent_info]:
                corewords_from_PubTerms[ent_info].append( term_cn )


    #
    return corewords_from_PubTerms


# 来自TmpTerms的 核心词实体名称表 tmp_terms
# 在没有语义标签时，存放到 tmp_terms 中
# 在具有语义标签后，存放到 pub_terms 中
def get_corewords_from_TmpTerms(db_info_dict, ssu_type_of_tags):
    # 求解目标
    # clean_name: [unclean_name]
    corewords_from_TmpTerms = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # build sql command
    sql_cmd = "select term_cn, term_cui, fully_confirmed, term_tui, term_sty from tmp_terms where mask_tag = 'N' " 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()      

    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   

    #
    for row in rows:        
        # 
        term_cn  = row[0]
        term_cui = row[1]
        fully_confirmed = row[2]

        # T901
        # Key Point in Medical Test
        ent_tui = row[3]
        ent_sty = row[4]

        # 核心词类型 
        # 默认 一般核心词
        core_type = '一般核心词'

        # 根据实际的 TUI 进行更改
        if ent_tui in ssu_type_of_tags:
            core_type = ssu_type_of_tags[ent_tui]


        # entity information as clean_name (对照 MRCONSO4CN 的 keywords_dict 进行处理)
        # ent_info = term_cui + '||' + term_tui + '||' + term_tag + '||' + core_type
        ent_info = ""


        # 使用 term_cn 占据 ent_cui 的位置 作为标准名称
        if True:
            # 
            std_term_str = term_cn
            #
            ent_info = std_term_str + '||' + ent_tui + '||' + ent_sty + '||' + core_type

        # 
        if ent_info not in corewords_from_TmpTerms:
            corewords_from_TmpTerms.setdefault( ent_info, [] )

        # 只考虑两个以上字符
        if len(term_cn) >= 2:
            if term_cn not in corewords_from_TmpTerms[ent_info]:
                corewords_from_TmpTerms[ent_info].append( term_cn )


    #
    return corewords_from_TmpTerms



# 获取 term_str 的 confirmed_cui
# [备忘] 之后可能还要引入 tmp_terms 或 umls_terms 的数据
#        好在这三张表结构一样，可以简化些操作
def get_confirmed_cui_of_terms( db_info_dict ):
    # 求解目标
    dict_of_terms_with_confirmed_cui = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    # [1] pub_terms
    # build sql command
    sql_cmd = "select term_cn, term_cui, fully_confirmed from pub_terms where mask_tag = 'N' " 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows_of_pub_terms = db_cursor.fetchall()    

    # 
    for row in rows_of_pub_terms:        
        # 
        term_cn  = row[0]
        term_cui = row[1]
        fully_confirmed = row[2]

        # 
        if fully_confirmed == 'Y':
            dict_of_terms_with_confirmed_cui.setdefault( term_cn, term_cui)


    # [2] tmp_terms
    # build sql command
    sql_cmd = "select term_cn, term_cui, fully_confirmed from tmp_terms where mask_tag = 'N' " 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows_of_tmp_terms = db_cursor.fetchall()    

    # 
    for row in rows_of_tmp_terms:        
        # 
        term_cn  = row[0]
        term_cui = row[1]
        fully_confirmed = row[2]

        # 
        if fully_confirmed == 'Y':
            dict_of_terms_with_confirmed_cui.setdefault( term_cn, term_cui)


    # [3] umls_terms
    sql_cmd = "select term_cn, term_cui from umls_terms_filtered " 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows_of_umls_terms = db_cursor.fetchall()           

    # 
    for row in rows_of_umls_terms:        
        # 
        term_cn  = row[0]
        term_cui = row[1]

        # umls terms 的CUI是确定无疑正确
        dict_of_terms_with_confirmed_cui.setdefault( term_cn, term_cui)


    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   


    #
    return dict_of_terms_with_confirmed_cui


# 来自 UMLS_Terms_Filter 的 核心词实体名称表 
# 则用 pub_syns 中的知识进行标准化处理
def get_corewords_from_UMLSTermsFiltered(db_info_dict, ssu_type_of_tags):
    # 求解目标
    # clean_name: [unclean_name]
    corewords_from_UMLSTermsFiltered = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # build sql command
    sql_cmd = "select term_cn, term_cui, term_tui, term_sty from umls_terms_filtered where mask_tag = 'N' " 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()      

    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   

    #
    for row in rows:        
        # 
        term_cn  = row[0]
        term_cui = row[1]

        #        
        ent_tui = row[2]
        ent_sty = row[3]

        # 核心词类型 
        # 默认 一般核心词
        core_type = '一般核心词'

        # 根据实际的 TUI 进行更改
        if ent_tui in ssu_type_of_tags:
            core_type = ssu_type_of_tags[ent_tui]


        # entity information as clean_name (对照 MRCONSO4CN 的 keywords_dict 进行处理)
        # ent_info = term_cui + '||' + term_tui + '||' + term_tag + '||' + core_type
        ent_info = ""

        # 如果 CUI 是确定正确的，使用 term_cui 作为标准名称
        # 这里的CUI必然是确定正确
        ent_info = term_cui + '||' + ent_tui + '||' + ent_sty + '||' + core_type


        # 
        if ent_info not in corewords_from_UMLSTermsFiltered:
            corewords_from_UMLSTermsFiltered.setdefault( ent_info, [] )

        # 只考虑两个以上字符
        if len(term_cn) >= 2:
            if term_cn not in corewords_from_UMLSTermsFiltered[ent_info]:
                corewords_from_UMLSTermsFiltered[ent_info].append( term_cn )


    #
    return corewords_from_UMLSTermsFiltered


# 来自 UMLS/SNOMED-CT 的 概念间的层级关系资源
def get_term_knowledge_from_UMLS_Hiers( db_info_dict ):
    # 
    child_cui_to_parent_cuis = {}
    parent_cui_to_child_cuis = {}

    #
    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # build sql command
    sql_cmd = "select CUI, PCUI, RELA from umls_hiers_filtered" 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()      

    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   

    #
    for row in rows:  
        term_cui = row[0]
        term_pcui = row[1]
        rel_type = row[2]

        if rel_type == 'isa':
            # 
            if term_cui not in child_cui_to_parent_cuis:
                child_cui_to_parent_cuis.setdefault( term_cui, set() )

            child_cui_to_parent_cuis[term_cui].add( term_pcui )

            #
            if term_pcui not in parent_cui_to_child_cuis:
                parent_cui_to_child_cuis.setdefault( term_pcui, set() )

            parent_cui_to_child_cuis[term_pcui].add( term_cui )

    #
    return [child_cui_to_parent_cuis, parent_cui_to_child_cuis]




# # 来自pub_syns 的 术语同义词、术语子结点知识表 pub_syns
# def get_term_knowledge_from_PubSyns( db_info_dict, dict_of_terms_with_confirmed_cui ):
#     # 求解目标
#     # clean_name: [unclean_name]
#     term_knowledge_from_PubSyns = {}

#     # 数据库信息
#     host = db_info_dict["host"]
#     port = db_info_dict["port"]
#     user = db_info_dict["user"]
#     passwd = db_info_dict["passwd"]
#     db = db_info_dict["db"]
#     charset = db_info_dict["charset"]

#     # 建立数据库连接 
#     db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
#     db_cursor = db_connection.cursor()    

#     #    
#     # build sql command
#     sql_cmd = "select term_prefered_str, synonym_terms, children_prefered_terms from pub_syns" 
#     # run sql command
#     db_cursor.execute(sql_cmd)
#     # get sql results
#     rows = db_cursor.fetchall()      

#     # 关闭数据库, 关闭游标
#     db_cursor.close()
#     db_connection.close()   

#     #
#     for row in rows:  
#         term_prefered_str = row[0]

#         #
#         list_of_synonym_terms = []

#         if row[1] not in [None, ""]:
#             list_of_synonym_terms = row[1].split('||')


#         # 观察 首选术语 是否能 CUI化
#         term_prefered_cui = ""

#         term_str_list = list_of_synonym_terms + [term_prefered_str]

#         for term_str in term_str_list:
#             if term_str in dict_of_terms_with_confirmed_cui:
#                 term_prefered_cui = dict_of_terms_with_confirmed_cui[term_str]
#                 break

#         # 如果可以CUI化，用 term_prefered_cui 代替 term_prefered_str
#         if term_prefered_cui != "":
#             term_prefered_str = term_prefered_cui

#         # 类似的, 将 list_of_synonym_terms 也进行 CUI化处理
#         # 混合式的存储方式
#         updated_list_of_synonym_terms = []

#         for term_str in list_of_synonym_terms:
#             if term_str in dict_of_terms_with_confirmed_cui:
#                 term_cui = dict_of_terms_with_confirmed_cui[term_str]
#                 updated_list_of_synonym_terms.append(term_cui)
#             else:
#                 updated_list_of_synonym_terms.append(term_str)


#         # 
#         list_of_child_terms = []

#         if row[2] not in [None, ""]:
#             list_of_child_terms = row[2].split("||")

#         # 类似的, 将 list_of_child_terms 也进行 CUI化处理
#         # 混合式的存储方式
#         updated_list_of_child_terms = []

#         for term_str in list_of_child_terms:
#             if term_str in dict_of_terms_with_confirmed_cui:
#                 term_cui = dict_of_terms_with_confirmed_cui[term_str]
#                 updated_list_of_child_terms.append(term_cui)
#             else:
#                 updated_list_of_child_terms.append(term_str)        


#         #
#         tmpinfo = {}
#         tmpinfo.setdefault( "syn_info",   updated_list_of_synonym_terms )
#         tmpinfo.setdefault( "child_info", updated_list_of_child_terms )

#         #
#         if term_prefered_str not in term_knowledge_from_PubSyns:
#             term_knowledge_from_PubSyns.setdefault( term_prefered_str, tmpinfo )
#         # 如果 term_prefered_str 已经存在了
#         else:
#             # 更新 (合并) 已经存在的 syn_info 和 child_info 信息
#             combined_synonym_terms  = term_knowledge_from_PubSyns[term_prefered_str]["syn_info"]   + updated_list_of_synonym_terms
#             combined_children_terms = term_knowledge_from_PubSyns[term_prefered_str]["child_info"] + updated_list_of_child_terms

#             #
#             combined_synonym_terms = list( set(combined_synonym_terms) )
#             combined_children_terms = list( set(combined_children_terms) )

#             # 更新
#             term_knowledge_from_PubSyns[term_prefered_str]["syn_info"]   = combined_synonym_terms
#             term_knowledge_from_PubSyns[term_prefered_str]["child_info"] = combined_children_terms

#     #
#     # print( "来自PubSyns的同义词和父子词", term_knowledge_from_PubSyns['C0015967'] )

#     #
#     return term_knowledge_from_PubSyns


# 来自 pub_syns 的 术语同义词、术语子结点知识表 pub_syns
# 只读取 term_str 信息
def get_term_knowledge_from_PubSyns( db_info_dict ):
    # 求解目标
    # clean_name: [unclean_name]
    term_knowledge_from_PubSyns = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]

    # 建立数据库连接
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()

    #
    # build sql command
    sql_cmd = "select term_prefered_str, synonym_terms, children_prefered_terms from pub_syns" 
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()

    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()

    #
    for row in rows:  
        term_prefered_str = row[0]

        #
        list_of_synonym_terms = []

        if row[1] not in [None, ""]:
            list_of_synonym_terms = row[1].split('||')


        # 
        list_of_child_terms = []

        if row[2] not in [None, ""]:
            list_of_child_terms = row[2].split("||")


        #
        tmpinfo = {}
        tmpinfo.setdefault( "syn_info",   list_of_synonym_terms )
        tmpinfo.setdefault( "child_info", list_of_child_terms )

        #
        if term_prefered_str not in term_knowledge_from_PubSyns:
            term_knowledge_from_PubSyns.setdefault( term_prefered_str, tmpinfo )
        # 如果 term_prefered_str 已经存在了
        else:
            # 更新 (合并) 已经存在的 syn_info 和 child_info 信息
            combined_synonym_terms  = term_knowledge_from_PubSyns[term_prefered_str]["syn_info"]   + list_of_synonym_terms
            combined_children_terms = term_knowledge_from_PubSyns[term_prefered_str]["child_info"] + list_of_child_terms

            #
            combined_synonym_terms = list( set(combined_synonym_terms) )
            combined_children_terms = list( set(combined_children_terms) )

            # 更新
            term_knowledge_from_PubSyns[term_prefered_str]["syn_info"]   = combined_synonym_terms
            term_knowledge_from_PubSyns[term_prefered_str]["child_info"] = combined_children_terms

    #
    # print( "来自PubSyns的同义词和父子词", term_knowledge_from_PubSyns['C0015967'] )

    #
    return term_knowledge_from_PubSyns



# 获取某中文术语字符串的 首选名称
def get_prefered_name_from_PubSyns( term_knowledge_from_PubSyns ):
    #
    prefered_name_from_PubSyns = {}

    #
    for term_prefered_str in term_knowledge_from_PubSyns:
        #
        prefered_name_from_PubSyns.setdefault( term_prefered_str, term_prefered_str )

        #
        list_of_synonym_terms = term_knowledge_from_PubSyns[term_prefered_str]['syn_info']

        #
        for syn_term in list_of_synonym_terms:
            prefered_name_from_PubSyns.setdefault( syn_term, term_prefered_str )

    #
    return prefered_name_from_PubSyns


# 增加一个术语的标准化函数，如果有CUI，标准化到CUI
# 如果没有CUI，有预设的同义词，标准化到首选词
def normalize_term_to_cui_or_prefer( term_str, dict_of_terms_with_confirmed_cui, prefered_name_from_PubSyns ):
    #
    normalized_term_str = term_str

    if term_str in dict_of_terms_with_confirmed_cui:
        normalized_term_str= dict_of_terms_with_confirmed_cui[term_str]
    else:
        if term_str in prefered_name_from_PubSyns:
            normalized_term_str= prefered_name_from_PubSyns[term_str]

    #
    return normalized_term_str 



# 获取某中文术语字符串的 子级术语
# term: [ list_of_child_terms ]
# 修改: 如果 term 有 CUI 
# CUI ： [list_of_child_terms + list_of_child_CUIs]
def get_children_terms_from_PubSyns( term_knowledge_from_PubSyns, dict_of_terms_with_confirmed_cui ):
    #
    children_terms_from_PubSyns = {}

    #
    for term_prefered_str in term_knowledge_from_PubSyns:
        #
        set_of_child_terms_as_str = set( term_knowledge_from_PubSyns[term_prefered_str]['child_info'] )

        # 
        set_of_child_terms_as_cui = set()

        for child_term_str in set_of_child_terms_as_str:
            if child_term_str in dict_of_terms_with_confirmed_cui:
                child_term_cui = dict_of_terms_with_confirmed_cui[child_term_str]
                # 
                set_of_child_terms_as_cui.add( child_term_cui ) 

        # 合并集合并记录
        combined_set_of_children_terms = set_of_child_terms_as_cui | set_of_child_terms_as_str

        children_terms_from_PubSyns.setdefault( term_prefered_str, combined_set_of_children_terms )

        # 如果 term_prefered_str 也有 CUI
        if term_prefered_str in dict_of_terms_with_confirmed_cui:
            term_prefered_cui = dict_of_terms_with_confirmed_cui[term_prefered_str]
            # 加入 children_terms_from_PubSyns 作为 key
            children_terms_from_PubSyns.setdefault( term_prefered_cui, combined_set_of_children_terms )

    #
    return children_terms_from_PubSyns


# 获取某中文术语字符串的 父级术语
# term: []      可以有多个父结点
# 类似的，STR 和 CUI 并存
def get_parent_terms_from_PubSyns( term_knowledge_from_PubSyns, dict_of_terms_with_confirmed_cui ):
    #
    parent_terms_from_PubSyns = {}

    #
    for term_prefered_str in term_knowledge_from_PubSyns:
        #
        set_of_child_terms_as_str = set( term_knowledge_from_PubSyns[term_prefered_str]['child_info'] )

        #
        for child_term_str in set_of_child_terms_as_str:
            #
            if child_term_str not in parent_terms_from_PubSyns:
                parent_terms_from_PubSyns.setdefault( child_term_str, set() )

            # 改成 set 类型
            parent_terms_from_PubSyns[child_term_str].add( term_prefered_str )

            # 加入父类的CUI
            if term_prefered_str in dict_of_terms_with_confirmed_cui:
                term_prefered_cui = dict_of_terms_with_confirmed_cui[term_prefered_str]
                #
                parent_terms_from_PubSyns[child_term_str].add( term_prefered_cui )


            # 加入 子类的 CUI做 key， 如果有
            if child_term_str in dict_of_terms_with_confirmed_cui:
                child_term_cui = dict_of_terms_with_confirmed_cui[child_term_str] 
                #
                if child_term_cui not in parent_terms_from_PubSyns:
                    parent_terms_from_PubSyns.setdefault( child_term_cui, set() )

                #
                parent_terms_from_PubSyns[child_term_cui].add( term_prefered_str )

                # 加入父类的CUI
                if term_prefered_str in dict_of_terms_with_confirmed_cui:
                    term_prefered_cui = dict_of_terms_with_confirmed_cui[term_prefered_str]    
                    #
                    parent_terms_from_PubSyns[child_term_cui].add( term_prefered_cui )                


    #
    return parent_terms_from_PubSyns


# 获取 术语的 原子化拆分 信息
def get_term_splits_info( db_info_dict ):
    #
    dict_of_term_splits_info = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]    

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #
    # build sql command
    sql_cmd = "select term_str, split_info from pub_splits "
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()  

    # 术语，术语拆分信息
    for row in rows:
        term_str = row[0]
        #
        split_info = row[1]
        term_parts = split_info.split('+')
        #
        dict_of_term_splits_info.setdefault( term_str, term_parts )


    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   


    #
    return dict_of_term_splits_info



# 获取 术语的 N联征组合 信息
def get_term_triad_info( db_info_dict ):
    #
    dict_of_triad_term_info = {}

    # 数据库信息
    host = db_info_dict["host"]
    port = db_info_dict["port"]
    user = db_info_dict["user"]
    passwd = db_info_dict["passwd"]
    db = db_info_dict["db"]
    charset = db_info_dict["charset"]    

    # 建立数据库连接 
    db_connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)
    db_cursor = db_connection.cursor()    

    #    
    # build sql command
    sql_cmd = "select composed_term_str, composed_term_syns, related_symptom_strs from pub_triads "
    # run sql command
    db_cursor.execute(sql_cmd)
    # get sql results
    rows = db_cursor.fetchall()  

    # 术语，术语拆分信息
    for row in rows:
        composed_term_str = row[0]
        composed_term_syns = row[1]
        related_symptom_strs = row[2]

        # 
        list_of_related_symptoms = []

        if related_symptom_strs not in [None, ""]:
            list_of_related_symptoms = related_symptom_strs.split("||")

        #
        if len(composed_term_str) >= 2:
            dict_of_triad_term_info.setdefault( composed_term_str, list_of_related_symptoms )

        # 
        if composed_term_syns not in [None, ""]:
            if len( composed_term_syns ) >=2 :
                list_of_composed_terms = composed_term_syns.split('||')
                for composed_term_str in list_of_composed_terms:
                    dict_of_triad_term_info.setdefault( composed_term_str, list_of_related_symptoms )

    # 关闭数据库, 关闭游标
    db_cursor.close()
    db_connection.close()   

    #
    return dict_of_triad_term_info



# 3.1 获取SSU核心词关联的属性词
# 关联方法: 近邻关联规则或作用范围关联规则的变种
# 具体方法: 确定非一般核心词如表型核心词的SCOPE, 把 SCOPE[0] 和 SCOPE[1] 放入到 attr_words_found 列表
# 确定 SCOPE[0] 和 SCOPE[1] 在 attr_words_found 中的位置, 截取SCOPE区间内的属性词，与核心词进行关联
def get_ent_related_attr_words(text_str, core_words_found, attr_words_found, info_of_ssu_attributes):
    # 求解目标
    # [(core_word_id, "has_attribute", attr_word_id)]
    ent_related_attr_words = []
    # 遍历 core_words_found, 获取SSU核心词 (参与语义结构单元形成的核心词)
    for core_word_idx, core_word_info in enumerate(core_words_found):
        #
        span_info, span_start, span_end = core_word_info


        # 核心词的类型 (表型核心词)
        core_type = span_info.split('||')[-1]

        # ('C0012833||T184||Sign or Symptom||表型核心词', 0, 1) 
        # 如果是SSU核心词 (参与语义结构单元形成的核心词)
        # 表型核心词
        # 部位核心词
        # 测量核心词
        # 药物核心词
        if '一般核心词' not in span_info:
            #
            # 核心词的作用范围 (逗号分割的子句)
            # 将核心词作用范围的求解封装为函数
            boundary_start, boundary_end = get_context_boundary_of_core_word( text_str, core_words_found, core_word_idx )
            # print("\n boundary_start and boundary_start ", span_info, boundary_start, boundary_end)


            # 确定 SCOPE[0] 和 SCOPE[1] 在 attr_words_found 中的位置, 截取SCOPE区间内的属性词，与核心词进行关联
            # [('轻度||严重程度||表型核心词', 3, 4)] 
            # 当 boundary_end 是 len(text_str) - 1时, boundary_end 设置为 boundary_end+1
            boundary_start_item = ( boundary_start, "boundary_start" )
            boundary_end_item   = ( boundary_end+1,   "boundary_end" )

            # 
            tmplist = [] 
            tmplist.append( boundary_start_item )
            tmplist.append( boundary_end_item )

            # [('轻度||严重程度||表型核心词', 3, 4)] 
            for attr_word_info in attr_words_found:
                # 属性词类型
                attr_type = attr_word_info[0].split("||")[-1]
                # 只考虑能匹配核心词类型的属性词
                if attr_type == core_type:                    
                    tmplist.append( (attr_word_info[1], attr_word_info ) )

            # sort the tuple list by the first element
            # print("tmplist ", tmplist)
            tmplist = sorted( tmplist, key=lambda x: x[0] )

            # [(0, 'boundary_start'), (3, ('轻度||严重程度||表型核心词', 3, 4)), (4, 'boundary_end')]
            # print("中间过程 output: tmplist ", tmplist)            

            # 
            boundary_start_item_idx = tmplist.index( boundary_start_item )
            boundary_end_item_idx   = tmplist.index( boundary_end_item )


            # 截取SCOPE区间内的属性词，与核心词进行关联
            attr_words_in_scope = tmplist[boundary_start_item_idx+1:boundary_end_item_idx]

            if len( attr_words_in_scope ) != 0:
                # 核心词 id span_start, span_end
                core_word_id = str( span_start ) + ':' + str( span_end )

                # [ADD] 避免添加不支持多个取值的属性
                existed_attributes = []

                for item_start, attr_word_info in attr_words_in_scope:
                    # attr_word_info [('轻度||严重程度||表型核心词', 3, 4)] 

                    attr_word_id = str( attr_word_info[1] ) + ':' + str( attr_word_info[2] )

                    # 
                    attr_cn_name = attr_word_info[0].split("||")[1]
                    allow_multi_values = True

                    if attr_cn_name in info_of_ssu_attributes:
                        if info_of_ssu_attributes[attr_cn_name]['ALLOW_MULTI_VALUE'] == '否':
                            allow_multi_values = False

                    # 
                    rel_name_str = 'has_attribute' 

                    # 如果该属性只能取一个值，且该核心词已经关联过该属性了，跳过
                    if allow_multi_values == False and attr_cn_name in existed_attributes:
                        continue
                    else:
                        # store the relation triple
                        ent_related_attr_words.append( (core_word_id, rel_name_str, attr_word_id) )
                        existed_attributes.append( attr_cn_name )

    #
    return ent_related_attr_words



# 3.2 获取SSU核心词关联的核心词
# 主要是 表型核心词 和 部位核心词之间的关联
def get_ent_related_core_words(text_str, core_words_found, dict_of_terms_with_confirmed_cui, prefered_name_from_PubSyns):
    # 求解目标
    # [(core_word_id, "locate_at", core_word_id)]
    ent_related_core_words = []


    # 获取部位核心词
    body_core_words_found = []
    phen_core_words_found = []


    # 遍历 core_words_found, 获取SSU中的表型核心词 (参与语义结构单元形成的核心词)
    for core_word_idx, core_word_info in enumerate(core_words_found):
        #
        span_info, span_start, span_end = core_word_info

        # 核心词的类型 (表型核心词)
        core_type = span_info.split('||')[-1]

        # ('C0012833||T184||Sign or Symptom||表型核心词', 0, 1) 
        # 如果是SSU核心词 (参与语义结构单元形成的核心词)
        # 表型核心词
        # 部位核心词
        # 测量核心词
        # 药物核心词
        if '表型核心词' in span_info:
            phen_core_words_found.append( core_word_info )
        elif '部位核心词' in span_info:
            
            # # 增加一个部位术语标准化的过程
            # body_span_info, body_span_start, body_span_end = core_word_info
            # body_core_word = body_span_info.split('||')[0]
            # body_core_word_info_std = normalize_term_to_cui_or_prefer( body_core_word, dict_of_terms_with_confirmed_cui, prefered_name_from_PubSyns )
            # body_span_info_std = body_span_info.replace(body_core_word, body_core_word_info_std)
            # # 重新形成tuple
            # core_word_info = (body_span_info_std, body_span_start, body_span_end)

            body_core_words_found.append( core_word_info )


    # 遍历 表型核心词，获取其范围内关联的部位核心词
    for core_word_idx, core_word_info in enumerate(phen_core_words_found):
        #
        span_info, span_start, span_end = core_word_info


        # 核心词的作用范围 (逗号分割的子句)
        # 将核心词作用范围的求解封装为函数
        # [Bug] 原来为 get_context_boundary_of_core_word( text_str, core_words_found, core_word_idx )
        #       但 core_word_idx 对应的是 phen_core_words_found 的索引
        boundary_start, boundary_end = get_context_boundary_of_core_word( text_str, phen_core_words_found, core_word_idx )

        #
        # if 'C0030252' in span_info:
            # print("\n boundary_start and boundary_start ", span_info, text_str[span_start-3:span_end+3], text_str[boundary_start:boundary_end+1])


        # 确定 SCOPE[0] 和 SCOPE[1] 在 attr_words_found 中的位置, 截取SCOPE区间内的属性词，与核心词进行关联
        # [('轻度||严重程度||表型核心词', 3, 4)] 
        # 当 boundary_end 是 len(text_str) - 1时, boundary_end 设置为 boundary_end+1
        boundary_start_item = ( boundary_start, "boundary_start" )
        boundary_end_item   = ( boundary_end+1,   "boundary_end" )
        # 
        tmplist = [] 
        tmplist.append( boundary_start_item )
        tmplist.append( boundary_end_item )


        # 加入部位核心词
        for body_word_info in body_core_words_found:
            # start, word_info
            tmplist.append( (body_word_info[1], body_word_info ) )

        # sort the tuple list by the first element
        # print("tmplist ", tmplist)
        tmplist = sorted( tmplist, key=lambda x: x[0] )

       
        # 
        boundary_start_item_idx = tmplist.index( boundary_start_item )
        boundary_end_item_idx   = tmplist.index( boundary_end_item )


        # 截取SCOPE区间内的部位词，与表型核心词进行关联
        body_words_in_scope = tmplist[boundary_start_item_idx+1:boundary_end_item_idx]
        if len( body_words_in_scope ) != 0:
            # 表型核心词 id span_start, span_end
            phen_core_word_id = str( span_start ) + ':' + str( span_end )

            #
            for item_start, body_word_info in body_words_in_scope:
                # [(core_word_id, "locate_at", core_word_id)]
                body_core_word_id = str( body_word_info[1] ) + ':' + str( body_word_info[2] )

                # 
                rel_name_str = 'locate_at' 

                # store the relation triple
                ent_related_core_words.append( (phen_core_word_id, rel_name_str, body_core_word_id) )

    #
    return ent_related_core_words



# 
# 核心词实体均需要标注
# 与核心词关联的属性词(术语型或模式型)实体需要标注 (但平时处于隐藏状态)
# 两段数据，后端数据写在函数中; 返回给前端的数据可写在接口中，函数之后，返回之前。
def get_annotated_ent_info( text_str, section_headers_found, core_words_found, attr_words_found, ent_related_attr_words ):
    # { ent_id: ent_info_dict }
    annotated_ent_info = {}

    # 纳入文本中扫描到的标题词
    for (span_info, span_start, span_end) in section_headers_found:
        # ent_info
        ent_info = {}

        # ent_id 
        ent_id = str( span_start ) + ':' + str( span_end )
        ent_info.setdefault( "ent_id", ent_id )

        # pos
        ent_info.setdefault( "span_start", span_start )
        ent_info.setdefault( "span_end",    span_end )

        # term_type
        # 标题词 T, 核心词 C, 属性词 A
        ent_info.setdefault( "term_type",  "标题词" )

        #
        ent_info.setdefault( "term_str", text_str[span_start:span_end+1] )  
        
        # 标题可作为"标题核心词", 不构成SSU的核心词    
        ent_info.setdefault( "core_type",  "标题核心词" )

        # 标准值
        # "标题||无标准名称"
        std_value = span_info.split("||")[1]
        ent_info.setdefault( "std_value",  std_value )

        # store
        annotated_ent_info.setdefault(ent_id, ent_info)


    # 纳入文本中扫描到的核心词，包含一般核心词和SSU核心词
    for (span_info, span_start, span_end) in core_words_found:
        # ent_info
        ent_info = {}

        # ent_id 
        ent_id = str( span_start ) + ':' + str( span_end )
        ent_info.setdefault( "ent_id", ent_id )

        # pos
        ent_info.setdefault( "span_start", span_start )
        ent_info.setdefault( "span_end",    span_end )

        # term_type
        ent_info.setdefault( "term_type",  "核心词" )

        # term_str [not necessary]
        ent_info.setdefault( "term_str", text_str[span_start:span_end+1] )


        # info
        term_cui, term_tui, term_sty, core_type = span_info.split("||")
        #
        ent_info.setdefault( "term_cui", term_cui )
        # 根据不同的TUI设置不同的颜色
        ent_info.setdefault( "term_tui", term_tui )
        ent_info.setdefault( "term_sty", term_sty )

        # 核心词类型: 表型核心词 
        ent_info.setdefault( "core_type", core_type )

        # store
        annotated_ent_info.setdefault(ent_id, ent_info)
        

    
    # 纳入文本中扫描到的与核心词关联的属性词实体

    # 获取与核心词关联的属性词实体id  
    # Get attr_word_id from ent_related_attr_words
    # (core_word_id, rel_name_str, attr_word_id)
    active_attr_word_ids = [item[2] for item in ent_related_attr_words]


    # 纳入文本中扫描到的与核心词关联的属性词实体
    # 如果某属性词与核心词未关联，那么不纳入标注
    # [前处理][数据准备] 属性 -关联-> 核心词
    # ent_related_attr_words.append( (core_word_id, rel_name_str, attr_word_id) )
    attr_related_to_ent = {}

    for (core_word_id, rel_name_str, attr_word_id) in ent_related_attr_words:
        if rel_name_str == 'has_attribute':
            attr_related_to_ent.setdefault( attr_word_id, core_word_id )

    # print( "\n attr_related_to_ent ", attr_related_to_ent )

    # [正式处理] 纳入需要标注的属性
    # [('轻度||严重程度||表型核心词', 3, 4)] 
    for (span_info, span_start, span_end) in attr_words_found:
        # ent_info
        ent_info = {}

        # ent_id 
        ent_id = str( span_start ) + ':' + str( span_end )
        ent_info.setdefault( "ent_id", ent_id )

        # pos
        ent_info.setdefault( "span_start", span_start )
        ent_info.setdefault( "span_end",    span_end )

        # term_type
        ent_info.setdefault( "term_type",  "属性词" )        

        # term_str [not necessary]
        ent_info.setdefault( "term_str", text_str[span_start:span_end+1] )

        # info
        std_value, attr_name, attr_of = span_info.split("||")

        #
        ent_info.setdefault( "std_value", std_value )
        ent_info.setdefault( "attr_name", attr_name )
        # attr_of 表型核心词
        ent_info.setdefault( "attr_of", attr_of )


        # 该属性词是否关联到了某个核心词
        # 如果关联了才保留
        if ent_id in attr_related_to_ent:
            # store
            annotated_ent_info.setdefault(ent_id, ent_info)        


    # sort ent by order in text
    # 按 ent 在文本中出现的先后顺序进行排序
    tmplist = []

    for ent_id in annotated_ent_info:
        #
        ent_info = annotated_ent_info[ent_id]

        # ent pos
        span_start = ent_info["span_start"]

        # 
        tmplist.append( (span_start, ent_id, ent_info) )

    #
    tmplist = sorted( tmplist, key=lambda x: x[0] )

    #
    sorted_annotated_ent_info = {}

    for (span_start, ent_id, ent_info) in tmplist:
        sorted_annotated_ent_info.setdefault( ent_id, ent_info )


    #
    return sorted_annotated_ent_info


# [获取文本中需要标注的关系信息 rel_info]
# 实体包含的属性信息
# 实体关联的实体信息
def get_annotated_rel_info( ent_related_attr_words, ent_related_core_words ):
    #
    annotated_rel_info = {}


    # 实体包含的属性信息
    for (core_word_id, rel_name_str, attr_word_id) in ent_related_attr_words:
        #
        if core_word_id not in annotated_rel_info:
            annotated_rel_info.setdefault( core_word_id, [] )

        # 数据结构 中心实体-关联实体  
        # core_ent_id: [ (related_ent_id, "关系名称") ]
        annotated_rel_info[core_word_id].append( (rel_name_str, attr_word_id) )

    # 表型实体关联的部位实体信息
    for (core_word_id, rel_name_str, attr_word_id) in ent_related_core_words:
        #
        if core_word_id not in annotated_rel_info:
            annotated_rel_info.setdefault( core_word_id, [] )

        # 数据结构 中心实体-关联实体  
        # core_ent_id: [ (related_ent_id, "关系名称") ]
        annotated_rel_info[core_word_id].append( (rel_name_str, attr_word_id) )    


    #
    return annotated_rel_info


# 对文本进行section划分, 找到文本中标题或section名称对应的位置
# [('标题核心词', 0, 1)]
def get_section_headers_of_text( text_str, section_header_patterns ):
    #
    section_headers_found = []


    # 教科书中的section header模式
    # 这些都作为一级标题问题也不大

    # 第一节
    # 【概述】
    # 一、
    # 1.1
    list_of_header_patterns = []
    list_of_header_patterns.append( "第[一二三四五六七八九]节.{2,25}\n" )
    list_of_header_patterns.append( "【.{2,20}】\n" )
    list_of_header_patterns.append( "[一二三四五六七八九]、.{2,25}\n" )
    list_of_header_patterns.append( "[1-9]\.[1-9]\s[a-zA-Z\u4e00-\u9fa5]{2,20}[\u4e00-\u9fa5]\n" )

    # 
    pattern_of_section_header = '|'.join( list_of_header_patterns )


    #
    matches_of_section_header = re.finditer( pattern_of_section_header, text_str )

    for m in matches_of_section_header:
        if m:
            #
            span_start = m.start()
            span_end   = m.end()-2
            # 这是一种标题类型
            # 这种标题的标准值是, 如果没有标准值, 设为无标准名称
            span_info  = "标题||无标准名称"

            # [MEMO]
            # 对于病历，可能会涉及到 section 的标准化

            section_headers_found.append( (span_info, span_start, span_end) )


    # [ADD]
    # 电子病历中常用的section表达模式扫描
    # span_info = "标题||标准名称"
    # 两者不存在交集的情况下，不需要对 text_str 进行 mask 处理
    keywords_found = section_header_patterns.extract_keywords( text_str, span_info= True )
    # print("\n keywords_found ", keywords_found )
    
    for (span_info, span_start, span_end) in keywords_found:
        # span_info 以设置为 "标题||标准名称"

        # 对 \n 结尾的字符的特殊处理 (因为\n不可见)
        if '\n' in text_str[span_start:span_end+1]:
            #
            span_end = span_end-1

        # 
        section_headers_found.append( (span_info, span_start, span_end) )
             
    #
    return section_headers_found


# 为便于合并核心词的同类属性
# 更改 annotated_rel_info 的存储形式
# from core_ent_id: [ (related_ent_id, "关系名称") ]
# to   core_ent_id: {"关系名称":[], "关系名称":[]
# 【Bug】 原以为的 rel_name_str 其实是 has_attribute 和 locate_at 两种
def group_annotated_rel_info_by_rel_name( annotated_ent_info, annotated_rel_info ):
    #
    grouped_annotated_rel_info = {}

    for core_ent_id in annotated_rel_info:
        # 关系类型名称 (具有...属性，位于...部位)     属性词id
        for (rel_name_str, attr_word_id) in annotated_rel_info[core_ent_id]:
            #
            if core_ent_id not in grouped_annotated_rel_info:
                grouped_annotated_rel_info.setdefault( core_ent_id, {} )

            # 如果是 具有...属性 的关系对
            # 提取 attr_word_id 对应的 属性名称
            attr_name_str = ""

            if rel_name_str == 'has_attribute':
                if attr_word_id in annotated_ent_info:
                    attr_name_str = annotated_ent_info[attr_word_id]["attr_name"]
            # 如果是关联的发作部位
            elif rel_name_str == 'locate_at':
                attr_name_str = '发作部位'

            # core_ent_id: {"(关联)属性名称", []}
            # 存在情况 发作部位
            if attr_name_str not in grouped_annotated_rel_info[core_ent_id]:
                grouped_annotated_rel_info[core_ent_id].setdefault( attr_name_str, [] )

            # core_ent_id: {"关系名称", [related_ent_id]}
            grouped_annotated_rel_info[core_ent_id][attr_name_str].append( attr_word_id )


    # 对关系名称:[ent_id, ent_id] 中的ent_id进行排序
    for core_ent_id in grouped_annotated_rel_info:
        for attr_name_str in grouped_annotated_rel_info[core_ent_id]:
            # original 
            attr_word_ids = grouped_annotated_rel_info[core_ent_id][attr_name_str]

            # sort attr_word_ids by positions
            tmplist = []

            for attr_word_id in attr_word_ids:
                attr_word_start = int( attr_word_id.split(':')[0] )
                tmplist.append( (attr_word_start, attr_word_id) )

            tmplist = sorted( tmplist, key=lambda x: x[0] )

            # update
            attr_word_ids = [ item[1] for item in tmplist ]

            # store
            grouped_annotated_rel_info[core_ent_id][attr_name_str] = attr_word_ids

    #
    return grouped_annotated_rel_info


# 属性 ent_id --> 核心词 ent_id
def get_ent_ids_as_attributes( annotated_rel_info ):
    #
    ent_ids_as_attributes = {}

    # annotated_rel_info  {'5:6': [('has_attribute', '8:9')]}
    for core_ent_id in annotated_rel_info:
        # 关系名称     属性词id
        for rel_name_str, attr_word_id in annotated_rel_info[core_ent_id]:
            # 
            # print( attr_word_id, core_ent_id )
            ent_ids_as_attributes.setdefault(attr_word_id,  core_ent_id)

    return ent_ids_as_attributes



# 获取 text_a 的二元标准化表达集合
def convert_annotations_to_binary_expressions( annotated_info_dict ):
    # 求解目标
    # 核心词的CUI::属性词的标准值
    # { ent_id: set(), ent_id: set(),}
    dict_of_binary_expressions = {}

    # 获取标注数据
    annotated_ent_info = annotated_info_dict["annotated_ent_info"]
    annotated_rel_info = annotated_info_dict["annotated_rel_info"]


    #
    for ent_id in annotated_ent_info:
        #
        ent_info = annotated_ent_info[ent_id]

        # term_type 
        # 标题词，核心词，属性词
        # 主要考虑 核心词 C (一般核心词，构成SSU的非一般核心词)
        if ent_info['term_type'] == '核心词':
            # 求解其 二元标准化表达组合
            # 设置为列表，便于固定第1位为 std_core_value
            binary_exp_list = []

            # 核心词的标准化表达 (std_value_of_core)
            std_core_value = ent_info['term_cui']

            # 核心词的字符串
            raw_coreterm_str = ent_info['term_str']

            # 核心词类型
            core_term_type = ent_info['core_type']

            if core_term_type == '测量核心词':
                raw_coreterm_str = std_core_value
            else:
                raw_coreterm_str = raw_coreterm_str

            # 判断该核心词的存在情况是否是 "无"
            has_negation_modifier = False


            if ent_id in annotated_rel_info:
                for rel_name_str, attr_ent_id in annotated_rel_info[ent_id]:
                    # 根据 attr_ent_id 获取 attr_ent_info
                    attr_ent_info = annotated_ent_info[attr_ent_id]

                    # 获取 std_value
                    std_attr_value = ""

                    # 如果关联的实体是一个属性词
                    if attr_ent_info['term_type'] == '属性词':
                        std_attr_value = attr_ent_info['std_value']

                    # 如果是否定词
                    if std_attr_value == '不存在':
                        has_negation_modifier = True


            # 将核心词的标准化表达加入到集合
            # 修正: {'白细胞(WBC)', '白细胞(WBC)::偏高'} {'白细胞(WBC)', '白细胞(WBC)::偏低'}
            # 这是两个完全不一样的表型异常，不能比对在一起，'白细胞(WBC)'不能放入进来
            # 即测量核心词不加入
            # 部位核心词也不要单独比对在一起，但是一般核心词是需要比较的
            # if core_term_type not in ["测量核心词", '部位核心词', '一般核心词']:
            
            # if core_term_type not in ["测量核心词", '部位核心词']:
            #     # 如果是肯定词
            #     if has_negation_modifier == False:
            #         binary_exp_list.append( raw_coreterm_str )
            #     # 如果是否定词, 在后面加入


            # 如果该核心词还具有关联属性或关联实体
            # 组成二元标准化表达加入到集合
            if core_term_type == '测量核心词':
                if ent_id in annotated_rel_info:
                    for rel_name_str, attr_ent_id in annotated_rel_info[ent_id]:
                        # 根据 attr_ent_id 获取 attr_ent_info
                        attr_ent_info = annotated_ent_info[attr_ent_id]

                        # 获取 std_value
                        std_attr_value = ""

                        # 如果关联的实体是一个属性词
                        if attr_ent_info['term_type'] == '属性词':
                            std_attr_value = attr_ent_info['std_value']

                        # 如果关联的实体是一个核心词, 那么标准值用CUI
                        if attr_ent_info['term_type'] == '核心词':
                            std_attr_value = attr_ent_info['term_cui']

                        # 将 std_core_value 与 std_attr_value 组合 并加入到集合
                        # 肯定情况，正常操作
                        if has_negation_modifier == False:
                            # binary_exp_list.append( std_core_value + "::" + std_attr_value )
                            binary_exp_list.append(std_attr_value)
                        # 存在否定情况
                        else:
                            if std_attr_value == '不存在':
                                # binary_exp_list.append( std_core_value + "::" + std_attr_value )
                                binary_exp_list.append(std_attr_value)
                            else:
                                # binary_exp_list.append( std_core_value + "::" + std_attr_value  + "::" + "无" )
                                binary_exp_list.append(std_attr_value)
            else:
                if ent_id in annotated_rel_info:
                    for rel_name_str, attr_ent_id in annotated_rel_info[ent_id]:
                        # 根据 attr_ent_id 获取 attr_ent_info
                        attr_ent_info = annotated_ent_info[attr_ent_id]

                        # 获取 std_value
                        std_attr_value = ""

                        # 如果关联的实体是一个属性词
                        if attr_ent_info['term_type'] == '属性词':
                            std_attr_value = attr_ent_info['std_value']

                        # 如果关联的实体是一个核心词, 那么标准值用CUI
                        if attr_ent_info['term_type'] == '核心词':
                            std_attr_value = attr_ent_info['term_cui']

                        # 将 std_core_value 与 std_attr_value 组合 并加入到集合
                        # 肯定情况，正常操作
                        if has_negation_modifier == False:
                            # binary_exp_list.append( std_core_value + "::" + std_attr_value )
                            binary_exp_list.append(std_attr_value)
                        # 存在否定情况
                        else:
                            if std_attr_value == '不存在':
                                # binary_exp_list.append( std_core_value + "::" + std_attr_value )
                                binary_exp_list.append(std_attr_value)
                            else:
                                # binary_exp_list.append( std_core_value + "::" + std_attr_value  + "::" + "无" )
                                binary_exp_list.append(std_attr_value)              


            # store by ent_id, binary_exp_set, std_core_value
            tmpdict = {}
            tmpdict.setdefault("binary_exp_list", binary_exp_list)
            tmpdict.setdefault("raw_coreterm_str", raw_coreterm_str)

            dict_of_binary_expressions.setdefault( ent_id, tmpdict)

    return dict_of_binary_expressions


# yangtao
# 获取核心词所在位置id和对应属性值id以及标准取值的字典：{'5:6': {'0:1': '急性', '2:3': '轻度'}}
# 输出结果补充进属性值
# 找到属性词的标准取值和id
# 最终结果为一个大的列表里面是属性id和属性标准值的小字典
def core_attr_id_stdvalue(ent_info_in_text, rel_info_in_text, ent_id, binary_expressions_of_text):
    # 找到两个句子的id和标准取值并写入列表
    attr_value_list = []
    attr_id_list = []
    if ent_info_in_text[ent_id]['core_type'] == '表型核心词':
        if ent_id in rel_info_in_text:
            for attr_id in rel_info_in_text[ent_id]:
                attr_id_list.append(attr_id[1])
            for attrs_value in binary_expressions_of_text[ent_id]['binary_exp_list']:
                attr_value_list.append(attrs_value)
    if ent_info_in_text[ent_id]['core_type'] == '测量核心词':
        if ent_id in rel_info_in_text:
            attr_value_list.append(binary_expressions_of_text[ent_id]['binary_exp_list'][0])
            for attr_id in rel_info_in_text[ent_id]:
                attr_id_list.append(attr_id[1])
    return attr_value_list, attr_id_list
# 根据上面的函数生成的两个列表，找到匹配上的属性的id位置
def align_core_attr_id_stdvalue(core_attr_id_stdvalue_a, core_attr_id_stdvalue_b):
    attr_align_info = []
    # 如果其中一个没有属性，或者两个都没有属性
    # if core_attr_id_stdvalue_a != None :
    if len(core_attr_id_stdvalue_a[0]) == 0 or core_attr_id_stdvalue_b[0] == 0:
        attr_align_info.append([])
    if dict(Counter(core_attr_id_stdvalue_a[0])) == dict(Counter(core_attr_id_stdvalue_b[0])):
        # print('完全一致',ent_id_a,core_attr_id_stdvalue_a[1],core_attr_id_stdvalue_b[1],ent_id_b)
        attr_align_info.append([core_attr_id_stdvalue_a[1],core_attr_id_stdvalue_b[1]])
    else:
        if len( set(core_attr_id_stdvalue_a[0]) & set(core_attr_id_stdvalue_b[0]) ) != 0:
            # print('部分一致',ent_id_a,[core_attr_id_stdvalue_a[0],core_attr_id_stdvalue_b[0]],ent_id_b)
            # 找到相同属性取值的交集
            attr_match_list_a = []
            attr_match_list_b = []
            inter_stdvalue = list(set(core_attr_id_stdvalue_a[0]) & set(core_attr_id_stdvalue_b[0]))
            for std_value in inter_stdvalue:
                # 根据标准取值的交集找到id中的交集位置
                index_a = core_attr_id_stdvalue_a[0].index(std_value)
                index_b = core_attr_id_stdvalue_b[0].index(std_value)
                attr_match_list_a.append(core_attr_id_stdvalue_a[1][index_a])
                attr_match_list_b.append(core_attr_id_stdvalue_b[1][index_b])
                # print(ent_id_a,core_attr_id_stdvalue_a[1][index_a],core_attr_id_stdvalue_b[1][index_b],ent_id_b)
            attr_align_info.append([attr_match_list_a,attr_match_list_b])
        else:
            # print('无一致属性',ent_id_a,[],[],ent_id_b)
            attr_align_info.append([])
    return attr_align_info



# 观察是否存在N联征比对
# 观察 a 中是否出现了 N 联证术语，如果有的话，获取其关联的症状列表，对这些症状术语进行标准化，在b中搜索，当都存在时，记录比对信息
def find_triad_alignments_btw_texts( ent_info_in_text_a, ent_info_in_text_b, dict_of_triad_term_info, \
                                        dict_of_terms_with_confirmed_cui, prefered_name_from_PubSyns):
    #
    dict_of_triad_alignments_found = {}


    # ent_info_in_text_b 的核心词集合
    ent_set_of_text_b = set()
    ent_b_exp_set_new = set()
    
    for ent_poskey in ent_info_in_text_b:
        ent_info = ent_info_in_text_b[ent_poskey]
        # 
        ent_term_str = ent_info["term_str"]
        #
        ent_term_cui = ""
        if "term_cui" in ent_info:
            ent_term_cui = ent_info["term_cui"]
        #
        ent_set_of_text_b.add(ent_term_str)

        if ent_term_cui != "":
            ent_set_of_text_b.add(ent_term_cui)
            
    for ent_b_raw_core in ent_set_of_text_b:
        if ent_b_raw_core in prefered_name_from_PubSyns:
            ent_b_std_core_new = normalize_term_to_cui_or_prefer(ent_b_raw_core, dict_of_terms_with_confirmed_cui, prefered_name_from_PubSyns)
            for exp_seg_b in ent_set_of_text_b:
                exp_seg_b = exp_seg_b.replace( ent_b_raw_core, ent_b_std_core_new )
                ent_b_exp_set_new.add( exp_seg_b )
        else:
            ent_b_exp_set_new.add( ent_b_raw_core )
    # print('ent_b_exp_set_new', ent_b_exp_set_new)


    # print( "ent_set_of_text_b ", ent_set_of_text_b  )

    # 观察 a 中是否出现了 N 联证术语
    for ent_a_poskey in ent_info_in_text_a:
        ent_info = ent_info_in_text_a[ent_a_poskey]
        # 
        ent_term_str = ent_info["term_str"]

        #
        if ent_term_str in dict_of_triad_term_info:
            list_of_related_symptoms = dict_of_triad_term_info[ent_term_str]
            # print( "list_of_related_symptoms ", list_of_related_symptoms )
            # 对术语进行标准化处理
            set_of_related_symptoms_normalized = set()
            # print( "set_of_related_symptoms_normalized ", ent_term_str, set_of_related_symptoms_normalized )
            #
            dict_of_related_symptoms = {}

            #
            for symptom_str in list_of_related_symptoms:
                normalized_symptom_str = normalize_term_to_cui_or_prefer(symptom_str, dict_of_terms_with_confirmed_cui, prefered_name_from_PubSyns)
                set_of_related_symptoms_normalized.add( normalized_symptom_str )
                dict_of_related_symptoms.setdefault( symptom_str, normalized_symptom_str )

            # N联证的具体症状都包含在b中
            # print("set_of_related_symptoms_normalized ", set_of_related_symptoms_normalized, ent_b_exp_set_new)            
            if set_of_related_symptoms_normalized.issubset( ent_b_exp_set_new ):
                # 解析，生成比对结果
                dict_of_triad_alignments_found.setdefault( ent_a_poskey, [] )

                # 对于N联征关联的每个症状，在b中找到对应
                for ent_b_poskey in ent_info_in_text_b:
                    for symptom_str in list_of_related_symptoms:
                    # for ent_b_poskey in ent_info_in_text_b:
                        #
                        ent_b_info = ent_info_in_text_b[ent_b_poskey]
                        #
                        ent_b_term_str = ent_b_info["term_str"]

                        #
                        ent_b_term_cui = ""

                        if "term_cui" in ent_b_info:
                            ent_b_term_cui = ent_b_info["term_cui"]

                        #
                        # print(normalized_symptom_str, symptom_str, [ent_b_term_str, ent_b_term_cui])
                        if normalized_symptom_str in [ent_b_term_str, ent_b_term_cui] or symptom_str in [ ent_b_term_str, ent_b_term_cui]:
                            dict_of_triad_alignments_found[ent_a_poskey].append( ent_b_poskey )
                            # print( "output ", normalized_symptom_str, symptom_str )
                            break
                        
    #
    return dict_of_triad_alignments_found

# # 定义一个函数来判断属性集合的相似类别
def attribute_set_similar_judgement(ent_a_attr_list, ent_b_attr_list): 
    if len(ent_a_attr_list) == 0 and len(ent_b_attr_list) == 0:
        attr_set_similar_categories = '完全相等'
    elif len(ent_a_attr_list) == 0 and len(ent_b_attr_list) != 0:
        attr_set_similar_categories = '不相似'
    elif len(ent_a_attr_list) != 0 and len(ent_b_attr_list) == 0:
        attr_set_similar_categories = '不相似'
    else:
        attr_set_align_result = []
        for ent_a_attr in ent_a_attr_list:
            for ent_b_attr in ent_b_attr_list:
                if ent_a_attr == ent_b_attr:
                    result = '完全相等'
                else:
                    result = '不相似'
                attr_set_align_result.append(result)
        count = attr_set_align_result.count('不相似')
        notsim_per = (count / len(attr_set_align_result)) * 100
        if notsim_per == 100.0:
            attr_set_similar_categories = '不相似'
        elif notsim_per == 0.0:
            attr_set_similar_categories = '完全相等'
        else:
            attr_set_similar_categories = '部分相似'
    return attr_set_similar_categories


# 优化后的函数，使用预加载的模型和tokenizer
def Term_judging_similar_categories(terms_a, terms_b):
    batch_size = 32  # 适当调整批量大小
    results = []
    if terms_a == terms_b:
        results.append('完全相等')
    else:
        for i in range(0, len(terms_a), batch_size):
            batch_a = terms_a[i:i+batch_size]
            batch_b = terms_b[i:i+batch_size]
            inputs = tokenizer(batch_a, batch_b, padding=True, truncation=True, return_tensors="pt", max_length=128)
            with torch.no_grad():
                predictions = model(**inputs)
            similar_categories = predictions.logits.argmax(dim=-1).tolist()
            for category in similar_categories:
                if category == 2:
                    results.append('完全相等')
                elif category == 1:
                    results.append('部分相似')
                else:
                    results.append('不相似')
    return results

def ssu_similar_categories(ent_a_core_type, ent_b_core_type, core_term_similar_categories, attr_set_similar_categories):
    if ent_a_core_type == '测量核心词' and ent_b_core_type == '测量核心词':
        if core_term_similar_categories == '完全相等' and attr_set_similar_categories == '完全相等':
            align_info_btw_ents = ['完全相等']
        else:
            align_info_btw_ents = ['不相似']
    else:
        if core_term_similar_categories == '完全相等' and attr_set_similar_categories == '完全相等':
            align_info_btw_ents = ['完全相等']
        elif core_term_similar_categories == '完全相等' and attr_set_similar_categories != '完全相等':
            align_info_btw_ents = ['部分相似']
        elif core_term_similar_categories == '部分相似':
            align_info_btw_ents = ['部分相似']
        else:
            align_info_btw_ents = ['不相似']
    return align_info_btw_ents



# 升级文本中 药物类实体 的 Pattern类关联属性 阅读功能
# 针对 药物类核心词 的 给药频率(每日3次)、 每次给药剂量 (20mg)、 连续给药时间(连续5天) 等属性
# 每次给药剂量: 1~2mg, 47.5~190mg, 每次2片，每日一片
# 每日服药频率: 每天三次，一日三次，一天2次; 每晚一次； 每12小时一次; 每8小时一次
# 连续服药天数: 
def update_info_of_drug_ents( text_str, annotated_info_dict ):
    # 求解目标
    updated_annotated_info_dict = {}


    # 获取标注数据
    annotated_ent_info = annotated_info_dict["annotated_ent_info"]
    annotated_rel_info = annotated_info_dict["annotated_rel_info"] 


    # Note
    # 每次给药剂量 的标准表达:  float + unit ;  float-float + unit
    # 给药频率     的标准表达:
    # 连续服药天数 的标准表达:



    # 确定 药物核心词 的SCOPE 可能会用到下一个核心词的坐标
    list_of_ent_ids = []

    for ent_id in annotated_ent_info:
        ent_info = annotated_ent_info[ent_id]

        # 只记录核心词
        ent_type = ent_info["term_type"]

        if ent_type == '核心词':
            # 核心词的位置
            span_start = ent_info['span_start']
            span_end   = ent_info['span_end'] 
            
            # 
            list_of_ent_ids.append( (span_start, ent_id) )

    # sort the tuple list by the first element
    # 可通过 (span_start_of_phenotype_ent, ent_id) 来确定坐标
    sorted_list_of_ent_ids = sorted( list_of_ent_ids, key=lambda x: x[0] )


    #
    # 记录需要更新的信息
    # {ent_id: tmpinfo}
    info_to_be_updated = {}    


    # 
    # [正式扫描]
    for ent_id in annotated_ent_info:
        # 实体信息 
        ent_info = annotated_ent_info[ent_id]

        # 核心词的位置
        span_start_of_drug_ent = ent_info['span_start']
        span_end_of_drug_ent   = ent_info['span_end']

        # 是否是 drug_ent
        is_drug_ent = False

        # Title, Core, Graph, Atrribute
        # term_type = ent_info['term_type']
        # 如果核心词
        if ent_info['term_type'] == '核心词':
            # 核心词类型
            if ent_info['core_type'] == '药物核心词':
                is_drug_ent = True


        # 如果不是 药物核心词 ，那么直接看下一个
        if not is_drug_ent:
            continue
        # 如果是 药物核心词
        else:
            # 记录需要更新的信息
            tmpinfo = {}

            # 在 药物核心词 的作用范围内搜索 每次给药剂量 表达模式
            ent_info_of_drug_dose = search_ent_info_of_drug_dose( text_str, annotated_ent_info, ent_id, sorted_list_of_ent_ids )

            # 如果存在符合规则的 给药剂量 实体
            # 记录到需要更新的信息中
            if len( ent_info_of_drug_dose ) != 0:
                # 需要更新的信息
                tmpinfo.setdefault( "ent_info_of_drug_dose", ent_info_of_drug_dose )

                #
                info_to_be_updated.setdefault(ent_id, tmpinfo)

            # 
            # 在 药物核心词 的作用范围内搜索 给药频率 表达模式
            ent_info_of_drug_freq = search_ent_info_of_drug_freq( text_str, annotated_ent_info, ent_id, sorted_list_of_ent_ids )


            # 如果存在符合规则的 给药频率 实体
            # 记录到需要更新的信息中
            if len( ent_info_of_drug_freq ) != 0:
                # 需要更新的信息
                tmpinfo.setdefault( "ent_info_of_drug_freq", ent_info_of_drug_freq )

                #
                info_to_be_updated.setdefault(ent_id, tmpinfo)

            # 
            # 在 药物核心词 的作用范围内搜索 连用时间 表达模式
            ent_info_of_drug_duration = search_ent_info_of_drug_duration( text_str, annotated_ent_info, ent_id, sorted_list_of_ent_ids )


            # 如果存在符合规则的 连用时间 实体
            # 记录到需要更新的信息中
            if len( ent_info_of_drug_duration ) != 0:
                # 需要更新的信息
                tmpinfo.setdefault( "ent_info_of_drug_duration", ent_info_of_drug_duration )

                #
                info_to_be_updated.setdefault(ent_id, tmpinfo)




    # 根据 info_to_be_updated
    # 更新 annotated_ent_info 和 annotated_rel_info
    for ent_id in info_to_be_updated:
        # 还原待更新的信息
        tmpinfo = info_to_be_updated[ent_id]

        # 1. 如果 药物核心词 存在 给药剂量 实体
        if "ent_info_of_drug_dose" in tmpinfo:
            ent_info_of_drug_dose = tmpinfo["ent_info_of_drug_dose"]
            ent_id_of_drug_dose   = ent_info_of_drug_dose["ent_id"]                

            # 在 annotated_ent_info 中 添加 ent_info_of_drug_dose
            annotated_ent_info.setdefault( ent_id_of_drug_dose, ent_info_of_drug_dose )  

            # 更新 annotated_rel_info
            if ent_id in annotated_rel_info:
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_drug_dose) )
            else:
                annotated_rel_info.setdefault( ent_id, [] )
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_drug_dose) )    

        # 2. 如果 药物核心词 存在 给药频率 实体
        if "ent_info_of_drug_freq" in tmpinfo:
            ent_info_of_drug_freq = tmpinfo["ent_info_of_drug_freq"]
            ent_id_of_drug_freq   = ent_info_of_drug_freq["ent_id"]                

            # 在 annotated_ent_info 中 添加 ent_info_of_drug_freq
            annotated_ent_info.setdefault( ent_id_of_drug_freq, ent_info_of_drug_freq )  

            # 更新 annotated_rel_info
            if ent_id in annotated_rel_info:
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_drug_freq) )
            else:
                annotated_rel_info.setdefault( ent_id, [] )
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_drug_freq) )  


        # 3. 如果 药物核心词 存在 连用时间 实体
        if "ent_info_of_drug_duration" in tmpinfo:
            ent_info_of_drug_duration = tmpinfo["ent_info_of_drug_duration"]
            ent_id_of_drug_duration   = ent_info_of_drug_duration["ent_id"]                

            # 在 annotated_ent_info 中 添加 ent_info_of_drug_duration
            annotated_ent_info.setdefault( ent_id_of_drug_duration, ent_info_of_drug_duration )  

            # 更新 annotated_rel_info
            if ent_id in annotated_rel_info:
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_drug_duration) )
            else:
                annotated_rel_info.setdefault( ent_id, [] )
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_drug_duration) )  


    # 记录更新
    updated_annotated_info_dict.setdefault( "annotated_ent_info", annotated_ent_info )
    updated_annotated_info_dict.setdefault( "annotated_rel_info", annotated_rel_info )



    #
    return updated_annotated_info_dict



# 升级文本中 表型类实体 的 持续时间 阅读功能
# 升级文本中 表型类实体 的 既往存在 阅读功能
def update_info_of_phenotype_ents( text_str, annotated_info_dict ):
    # 求解目标
    updated_annotated_info_dict = {}

    # 获取标注数据
    annotated_ent_info = annotated_info_dict["annotated_ent_info"]
    annotated_rel_info = annotated_info_dict["annotated_rel_info"] 

    # 记录需要更新的信息
    # {ent_id: tmpinfo}
    info_to_be_updated = {}


    # [正式扫描]
    for ent_id in annotated_ent_info:
        # 实体信息 
        ent_info = annotated_ent_info[ent_id]

        # 核心词的位置
        span_start_of_phenotype_ent = ent_info['span_start']
        span_end_of_phenotype_ent   = ent_info['span_end']


        # 是否是 phenotype_ent
        is_phenotype_ent = False

        # Title, Core, Graph, Atrribute
        # term_type = ent_info['term_type']
        # 如果核心词
        if ent_info['term_type'] == '核心词':
            # 核心词类型
            if ent_info['core_type'] == '表型核心词':
                is_phenotype_ent = True

        # 如果不是 表型核心词 ，那么直接看下一个
        if not is_phenotype_ent:
            continue
        # 如果是表型核心词
        # 在 right context 中搜索 持续时间 属性实体
        # 如果能有效关联，对 持续时间 表达模式 标准化 
        # 10 年/月/周/天/小时/分钟/秒钟       
        else:
            # 需要更新的信息
            tmpinfo = {}

            # 尝试寻找 持续时间 实体 (函数化)(模块化)
            # 未来处理 扩展属性 更好
            ent_info_of_time_duration = search_ent_info_of_time_duration( text_str, annotated_ent_info, ent_id )

            # 如果存在符合规则的 持续时间 实体
            # 记录到需要更新的信息中
            if len( ent_info_of_time_duration ) != 0:
                tmpinfo.setdefault( "ent_info_of_time_duration", ent_info_of_time_duration )
                #
                info_to_be_updated.setdefault(ent_id, tmpinfo)

            #
            # 尝试寻找 既往存在 实体
            ent_info_of_past_presence = search_ent_info_of_past_presence( text_str, annotated_ent_info, ent_id )

            # 如果存在符合规则的 既往存在 实体
            # 记录到需要更新的信息中
            if len( ent_info_of_past_presence ) != 0:
                tmpinfo.setdefault( "ent_info_of_past_presence", ent_info_of_past_presence )
                #
                info_to_be_updated.setdefault(ent_id, tmpinfo)             



    # 根据 info_to_be_updated
    # 更新 annotated_ent_info 和 annotated_rel_info
    for ent_id in info_to_be_updated:
        # 还原待更新的信息
        tmpinfo = info_to_be_updated[ent_id]

        # 如果 表型核心词 存在 持续时间 属性信息
        if "ent_info_of_time_duration" in tmpinfo:
            ent_info_of_time_duration = tmpinfo["ent_info_of_time_duration"]
            ent_id_of_time_duration   = ent_info_of_time_duration["ent_id"]                

            # 在 annotated_ent_info 中 添加 ent_info_of_time_duration
            annotated_ent_info.setdefault( ent_id_of_time_duration, ent_info_of_time_duration )  

            # 更新 annotated_rel_info
            if ent_id in annotated_rel_info:
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_time_duration) )
            else:
                annotated_rel_info.setdefault( ent_id, [] )
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_time_duration) )    


        # 如果 表型核心词 存在 既往存在 属性信息
        # 存在情况 好像也可以取 mutiple values 
        # 无 高血压 病史 (先不用做改动)
        if "ent_info_of_past_presence" in tmpinfo:
            ent_info_of_past_presence = tmpinfo["ent_info_of_past_presence"]
            ent_id_of_past_presence   = ent_info_of_past_presence["ent_id"]                

            # 在 annotated_ent_info 中 添加 ent_info_of_past_presence
            annotated_ent_info.setdefault( ent_id_of_past_presence, ent_info_of_past_presence )  

            # 更新 annotated_rel_info
            if ent_id in annotated_rel_info:
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_past_presence) )
            else:
                annotated_rel_info.setdefault( ent_id, [] )
                annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_past_presence) )   



    # 记录更新
    updated_annotated_info_dict.setdefault( "annotated_ent_info", annotated_ent_info )
    updated_annotated_info_dict.setdefault( "annotated_rel_info", annotated_rel_info )

    #
    return updated_annotated_info_dict




# 进一步的扫描 annotated_info_dict 中的类型为 测量核心词 的 ent
# 如果该 测量核心词 在具有参考范围的可判读列表中
# 阅读该核心词的上下文，进行 数字 --> 分类的判读
# 如果能够进行判读，则 
# update ent_info (主要是在该ent中加入样本来源信息) (同时加入测量结果的ent)
# update rel_info (加入 测量核心词 ent 和 测量结果 ent 的关联信息)
def update_info_of_observable_ents( text_str, annotated_info_dict, sample_name_processor_cn, 
                                        mapping_of_observable_ent_names, dict_of_ref_ranges ):
    # 求解目标
    updated_annotated_info_dict = {}
    # consists of updated_annotated_ent_info, updated_annotated_rel_info


    # 获取标注数据
    annotated_ent_info = annotated_info_dict["annotated_ent_info"]
    annotated_rel_info = annotated_info_dict["annotated_rel_info"] 


    # 记录需要更新的信息
    # {ent_id: tmpinfo}
    info_to_be_updated = {}

    for ent_id in annotated_ent_info:
        # 实体信息 
        ent_info = annotated_ent_info[ent_id]

        # 测量核心词的位置
        span_start_of_observable_ent = ent_info['span_start']
        span_end_of_observable_ent   = ent_info['span_end']


        # 是否是 observable_ent
        is_observable_ent = False

        # Title, Core, Graph, Atrribute
        # term_type = ent_info['term_type']
        # 如果核心词
        if ent_info['term_type'] == '核心词':
            # 核心词类型
            if ent_info['core_type'] == '测量核心词':
                is_observable_ent = True
                # print( '\n observable_ent_str ', ent_info['term_str'] )

        # 如果不是 测量核心词 ，那么直接看下一个
        if not is_observable_ent:
            continue


        # 如果是测量核心词, 通过 拼接 测量核心词 @ 样本来源 观察数据库中是否收录了它的参考范围
        # 解析 flashtext 扫描该测量核心词产生的 clean_name 
        # 对于 来自LATTE 中的测量核心词，term_cui 使用 ent_first_name 来替代
        # 对于 来自UMLS  中的测量核心词，term_cui 使用 cui
        # 
        # 判断方法: 将测量核心词的 term_str 通过 mapping_of_observable_ent_names 标准化
        # 观察标准化后的名称是否在 dict_of_ref_ranges 中
        has_reference_range = False
        ent_str_normalized  = ""
        ent_ref_ranges = {}

        # 获取 测量核心词 的 样本来源 标准名称
        # 样本来源 default NULL
        sample_str_normalized = 'NULL'
        sample_cui            = "NULL"


        if is_observable_ent:
            # 获取 测量核心词 的 标准名称
            ent_str_in_text = ent_info['term_str'] 
            # 
            ent_str_normalized = ""
            # 
            if ent_str_in_text in mapping_of_observable_ent_names:
                ent_str_normalized = mapping_of_observable_ent_names[ent_str_in_text]

            # 扫描 left_context 中的样本来源
            boundary_start = max(span_start_of_observable_ent - 100, 0)

            # 定位 ent_span_start 之前的第一个分号或句号 在文本中所在的位置 
            shift_count = 0
            for char in text_str[0:span_start_of_observable_ent][::-1]:
                # 偏移计数
                shift_count += 1
                # 如果遇到下述符号, 即是边界
                if char in ';；。\n和': 
                    # 更新 boundary_start 位置
                    boundary_start = span_start_of_observable_ent - shift_count 

            # 扫描 boundary_start 到 span_start_of_observable_ent 之间的文本
            samples_found = sample_name_processor_cn.extract_keywords( text_str[boundary_start:span_start_of_observable_ent] )

            # 如果能扫描到样本名称
            # [NOTE] 若产生了 False Positive, 可考虑对 文本 进行 MASK 处理
            if len( samples_found ) != 0:
                # clean_name = "||".join(  [value_cn_name, value_info_dict["ATTR_CN_NAME"], value_info_dict["SSU_TYPE"]] ) 
                first_sample_info = samples_found[0]
                # 
                first_sample_name  = first_sample_info.split('||')[0]

                # 
                if first_sample_name != '血液':
                    sample_str_normalized = first_sample_name

            # 生成  ent_str_normalized @ sample_str_normalized 结构
            key_for_observable_ent = ent_str_normalized + '@' + sample_str_normalized


            # 查询 ent_str_normalized @ sample_str_normalized 是否在 dict_of_ref_ranges 中
            if key_for_observable_ent in dict_of_ref_ranges:
                has_reference_range = True
                ent_ref_ranges = dict_of_ref_ranges[key_for_observable_ent]

        # 如果该 测量核心词 在知识库中不具有参考范围，可跳过
        if not has_reference_range:
            continue


        # 如果是测量核心词，且在LATTE数据库中具有参考范围
        # 确定 测量核心词 的 right_context, 搜索其中的 allowed_units
        # 如果 有 allowed_unit, 搜索其中的 numeric pattern 
        # 如果 allowed_units 允许为空，测量核心词 和 numeric pattern 的表达应该有某种模式 (比如 X X ,或。)
        # 确定 测量核心词 的 left_context, 扫描其中的 样本来源 关键字
        if is_observable_ent and has_reference_range:
            # 尝试寻找 测量结果实体
            ent_of_measure_result = ""

            # 该 测量核心词 在知识库中支持判读的测量单位
            # 如果可以不用单位，那么 allowed_units 中会有 'NULL'
            # unit 进行了统一的小写预处理
            allowed_units = list( ent_ref_ranges.keys() )

            # 
            pattern_of_units = []
            # 测量单位允许为空
            allow_null_unit = False

            for allowed_unit in allowed_units:
                if allowed_unit != 'NULL':
                    # re.escape(pattern) 可以对字符串中所有可能被解释为正则运算符的字符进行转义
                    pattern_of_units.append( re.escape(allowed_unit) )
                else:
                    allow_null_unit = True

            
            pattern_of_units = '|'.join( pattern_of_units )


            # 
            ent_of_measure_unit = ""

            # 测量核心词的位置
            span_start_of_observable_ent = ent_info['span_start']
            span_end_of_observable_ent   = ent_info['span_end']


            # 测量单位
            # 搜索核心词的一定范围内的right_context 是否存在 allowed_units
            exist_allow_unit = True

            # 测量单位
            match_object_of_unit = re.search( pattern_of_units, 
                                    text_str[span_end_of_observable_ent+1: span_end_of_observable_ent+15], re.I )
            
            # 如果能搜索到 测量单位
            if match_object_of_unit:
                # 测量单位在文中的位置坐标
                span_start_of_measure_unit = match_object_of_unit.start() + span_end_of_observable_ent+1
                span_end_of_measure_unit   = match_object_of_unit.end()   + span_end_of_observable_ent+1

                # 如果 测量核心词 和 测量单位 之间有以下符号
                # 那么 exist_allow_unit 设置为 False
                for char in text_str[span_end_of_observable_ent: span_start_of_measure_unit]:
                    if char in ',，;；。\n':
                        exist_allow_unit = False
            # 如果搜索不到测量单位
            else:
                exist_allow_unit = False


            # 如果不能搜索到测量单位，且测量单位不能为空，可以跳过了
            if exist_allow_unit == False and allow_null_unit == False:
                continue

            # 如果能搜索到测量单位，记录测量单位的以下信息
            str_of_unit = ""
            span_start_of_measure_unit = ""
            span_end_of_measure_unit   = ""

            if match_object_of_unit:
                # 测量单位在文中的位置坐标
                span_start_of_measure_unit = match_object_of_unit.start() + span_end_of_observable_ent+1
                span_end_of_measure_unit   = match_object_of_unit.end()   + span_end_of_observable_ent+1
                # 
                str_of_unit = match_object_of_unit.group()   
                #
                # print("\n str_of_unit ", str_of_unit)
            else:
                # 如果搜索不到, 测量单位可为空，则设置为 NULL
                if allow_null_unit:
                    str_of_unit = 'NULL'         


            # 测量数值
            # 如果 exist_allow_unit = True
            # 查看 核心词和测量单位之间是否存在合法的value
            exist_allow_value = False

            str_of_value = ""
            span_start_of_measure_value = ""
            span_end_of_measure_value   = ""

            # 如果 存在 allowed_unit, 搜索 核心词 和 单位之间 的 测量数值 
            if exist_allow_unit:
                # float                 
                # pattern_of_float_numbers  = "\d*\.?\d+"
                # 血压表达式 140/110 (需要处理，因为太常见了） marker /
                # pattern_of_blood_pressure = "\d+/\d+"
                # ratio  1.0:160.0  marker :
                # pattern_of_ratio_numbers = "\d*\.?\d+[:：]\d*\.?\d+"

                # 
                pattern_of_values = '\d*\.?\d+[:：]\d*\.?\d+|\d+/\d+|\d*\.?\d+'

                # 搜索 核心词 和 单位之间 的数值模式
                # print( "test span_end_of_observable_ent ", span_end_of_observable_ent, type(span_start_of_measure_unit) )
                match_object_of_value = re.search( pattern_of_values, 
                                                    text_str[span_end_of_observable_ent+1: span_start_of_measure_unit] )

                # 如果可以搜索到 match_object_of_value
                if match_object_of_value:
                    # 测量数值在文本中的坐标
                    span_start_of_measure_value = match_object_of_value.start() + span_end_of_observable_ent+1
                    span_end_of_measure_value   = match_object_of_value.end()   + span_end_of_observable_ent+1 
                    str_of_value = match_object_of_value.group() 

                    # 测量数值和核心词之间的距离不能过远
                    # 如果过远，则认为是虚假的 测量数值 
                    if abs(span_start_of_measure_value - span_end_of_observable_ent) <= 5:
                        exist_allow_value = True
            # 如果 不存在 allowed_unit, 但 allowed_unit 可以为空
            # 搜索 +15 范围内的 数值模式
            else:
                #
                if allow_null_unit:
                    # 数值模式
                    pattern_of_values = '\d*\.?\d+[:：]\d*\.?\d+|\d+/\d+|\d*\.?\d+'

                    # 搜索 核心词 和 核心词 +15 的数值模式
                    match_object_of_value = re.search( pattern_of_values, 
                                                        text_str[span_end_of_observable_ent+1: span_end_of_observable_ent+15] )

                    # 如果可以搜索到 match_object_of_value
                    if match_object_of_value:
                        # 测量数值在文本中的坐标
                        span_start_of_measure_value = match_object_of_value.start() + span_end_of_observable_ent+1
                        span_end_of_measure_value   = match_object_of_value.end()   + span_end_of_observable_ent+1

                        # 测量数值和核心词之间的距离不能过远
                        # 如果过远，则认为是虚假的 测量数值                 
                        if abs(span_start_of_measure_value - span_end_of_observable_ent) <= 5:
                            exist_allow_value = True


            # 如果不存在合法的测量数值，那么也可以跳过了
            if not exist_allow_value:
                continue
            # else:
            #     print( "\n str_of_value ",  str_of_value)


            # 正常性判断
            # 如果 exist_allow_unit = True
            str_of_normality = ""

            # 需要 
            # str_of_unit 
            # str_of_value

            # 在存在 合法的 str_of_value 的情况下
            # 对 str_of_value 的 正常性 进行判读
            if exist_allow_value:
                # [NOTE]
                # 从文本中获取 gender, age 等信息
                # dict_of_demographic_info = get_demographic_info_from_text( text_str )

                # ent_ref_ranges
                # 根据测量单位 确定 要使用的参考范围
                # 需要注意，ent_ref_ranges 中的 单位进行了大写，所以这里提取时也需要大写
                # 加入 text_str 参数，以便从中识别 性别与年龄的关键词
                used_ref_range = get_used_ref_range( ent_ref_ranges, str_of_unit.upper(), text_str )
                # print( '\n used_ref_range ', used_ref_range )

                # 确定 正常范围 上限和下限
                lower_limit = ""
                upper_limit = ""

                if "lower_limit" in used_ref_range:
                    lower_limit = used_ref_range["lower_limit"]

                if "upper_limit" in used_ref_range:
                    upper_limit = used_ref_range['upper_limit']

                # 根据 str_of_value, lower_limit, upper_limit 判读 正常性
                str_of_normality = judge_normality_by_value( str_of_value, lower_limit, upper_limit )



            # 如果可判读 正常性
            # 记录该 测量核心词 需要添加的信息
            # 测量核心词 (测量对象) ent_id: sample_name, sample_cui; 
            # 测量结果   (属性词)   ent_id: 
            # 测量对象  (has_attr) 测量结果 
            if str_of_normality != "":
                # 记录要更新的信息
                tmpinfo = {}

                # ent_id
                # 观测对象的 ent_info 
                ent_info.setdefault( "sample_name", sample_str_normalized)
                ent_info.setdefault( "sample_cui",  sample_cui ) 

                # 测量结果的 ent_info 
                ent_info_of_measure_result = {}

                # 从测量数值开始 
                span_start_of_measure_result = span_start_of_measure_value

                # 
                span_end_of_measure_result = ""

                # 如果测量单位不为空
                if str_of_unit != 'NULL':
                    # 来自正则表达式的位置坐标需要减1
                    span_end_of_measure_result = span_end_of_measure_unit -1
                else:
                    # 如果测量单位为空, span_end 设置为测量结果的末尾
                    span_end_of_measure_result = span_end_of_measure_value -1

                # 
                ent_id_of_measure_result = str(span_start_of_measure_result) + ':' + str(span_end_of_measure_result)

                # term_type 属性词
                # term_str  text_str[span_end_of_measure_result:span_end_of_measure_result+1]
                # std_value str_of_normality
                # attr_name 结果判读 (所以并不需要测量结果这一属性，意味着无法判读的结果不会被标注)
                ent_info_of_measure_result.setdefault( "ent_id", ent_id_of_measure_result )
                ent_info_of_measure_result.setdefault( "span_start", span_start_of_measure_result)
                ent_info_of_measure_result.setdefault( "span_end",   span_end_of_measure_result)
                ent_info_of_measure_result.setdefault( "term_type", "属性词" )
                ent_info_of_measure_result.setdefault( "term_str",  text_str[span_start_of_measure_result:span_end_of_measure_result+1])
                ent_info_of_measure_result.setdefault( "std_value", str_of_normality )
                ent_info_of_measure_result.setdefault( "attr_name", "结果判读" )
                ent_info_of_measure_result.setdefault( "attr_of", "测量核心词" )

                # 需要更新的信息
                tmpinfo.setdefault("ent_info_of_observable_ent", ent_info)                
                tmpinfo.setdefault( "ent_info_of_measure_result", ent_info_of_measure_result )
                # 
                info_to_be_updated.setdefault(ent_id, tmpinfo)


    # 根据 info_to_be_updated
    # 更新 annotated_ent_info 和 annotated_rel_info
    for ent_id in info_to_be_updated:
        # 还原待更新的信息
        tmpinfo = info_to_be_updated[ent_id]

        ent_info_of_observable_ent = tmpinfo["ent_info_of_observable_ent"]
        ent_info_of_measure_result = tmpinfo["ent_info_of_measure_result"]
        ent_id_of_measure_result   = ent_info_of_measure_result["ent_id"]


        # 直接更新 ent_id 对应的 ent_info 
        annotated_ent_info[ent_id] = ent_info_of_observable_ent

        # 在 annotated_ent_info 中 添加 ent_info_of_measure_result
        annotated_ent_info.setdefault( ent_id_of_measure_result, ent_info_of_measure_result )

        # 更新 annotated_rel_info
        if ent_id in annotated_rel_info:
            annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_measure_result) )
        else:
            annotated_rel_info.setdefault( ent_id, [] )
            annotated_rel_info[ent_id].append( ("has_attribute", ent_id_of_measure_result) )


    # 记录更新
    updated_annotated_info_dict.setdefault( "annotated_ent_info", annotated_ent_info )
    updated_annotated_info_dict.setdefault( "annotated_rel_info", annotated_rel_info )


    #
    return updated_annotated_info_dict



# 根据 str_of_value, lower_limit, upper_limit 判读 正常性
def judge_normality_by_value( str_of_value, lower_limit, upper_limit ):
    # 
    str_of_normality = ""

    # 如果 str_of_value 是一个 float
    # can_be_judged = True

    #
    if str_of_value.replace('.','',1).isdigit():
        # 那么 lower_limit 和 upper_limit 也需要是一个实数
        if lower_limit.replace('.','',1).isdigit() and upper_limit.replace('.','',1).isdigit():
            #
            measure_value = float( str_of_value )
            lower_value   = float( lower_limit )
            uppper_value  = float( upper_limit )

            # 
            if measure_value >= lower_value and measure_value <= uppper_value:
                str_of_normality = '正常'
            elif measure_value < lower_value:
                str_of_normality = '偏低'
            elif measure_value > uppper_value:
                str_of_normality = '偏高'
    #  
    else:
        # 血压型或比值型，看后一位
        if re.search('/|:|：', str_of_value):
            #
            str_of_last_value = re.split('/|:|：', str_of_value)[-1]

            # lower_limit 和 upper_limit 也需要是同样的结构
            str_of_lower_limit = ""
            str_of_upper_limit = ""

            if re.search('/|:|：', lower_limit):
                str_of_lower_limit = re.split('/|:|：', lower_limit)[-1]

            #
            if re.search('/|:|：', upper_limit):
                str_of_upper_limit = re.split('/|:|：', upper_limit)[-1]

            #
            if str_of_last_value.replace('.','',1).isdigit():
                #
                if str_of_lower_limit.replace('.','',1).isdigit() and str_of_upper_limit.replace('.','',1).isdigit():
                    #
                    measure_value = float( str_of_last_value )
                    lower_value   = float( str_of_lower_limit )
                    uppper_value  = float( str_of_upper_limit )

                    # 
                    if measure_value >= lower_value and measure_value <= uppper_value:
                        str_of_normality = '正常'
                    elif measure_value < lower_value:
                        str_of_normality = '偏低'
                    elif measure_value > uppper_value:
                        str_of_normality = '偏高'                    


    #
    return str_of_normality                
            

# 判断一个字符串是否是实数
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


# 目前取该测量单位对应的第一组参考范围，不考虑性别和年龄
# 看来需要考虑性别与年龄呢
def get_used_ref_range( ent_ref_ranges, str_of_unit, text_str ):
    #
    used_ref_range = {}

    # 考虑性别与年龄
    # 默认: 不需要考虑性别、年龄，
    # 如果没有扫描到文本中的年龄信息，年龄设置为 20 岁，男性
    gender_code_of_patient = '男'
    age_num_of_patient     = 20
    age_unit_of_patient    = '岁'

    # 扫描性别
    if re.search("女|女性|妇|子宫|卵巢", text_str):
        gender_code_of_patient = '女'

    if re.search("男|男性|睾丸|阴茎|龟头", text_str):
        gender_code_of_patient = '男'        

    # 扫描年龄 暂时不考虑多少天、多少周
    if re.search('\d+.{0,3}岁', text_str):
        # 解析数字
        age_snippet = re.search('\d+.{0,3}岁', text_str).group()
        age_num_of_patient     = float( re.search( '\d+', age_snippet ).group() )
        age_unit_of_patient    = '岁'

    
    #
    if str_of_unit in ent_ref_ranges:
        # 如果只有1组参考范围，直接使用该组参考范围，只要单位对得上
        if len( ent_ref_ranges[str_of_unit] ) == 1:
            used_ref_range = ent_ref_ranges[str_of_unit][0]
        # 如果有多组参考范围
        elif len( ent_ref_ranges[str_of_unit] ) > 1:
            # 需要考虑参考范围的使用性别和年龄
            selected_proper_range = False

            for tmpinfo in ent_ref_ranges[str_of_unit]:
                # 参考范围的适用性别
                # 如果不需要考虑性别，此值为空
                gender_code_of_ref = tmpinfo["gender_code"]


                # 参考范围的适用年龄
                # 如果不需要考虑年龄，此值为空
                lower_age_of_ref = tmpinfo["lower_age"]
                upper_age_of_ref = tmpinfo["upper_age"]
                age_unit_of_ref  = tmpinfo["unit_of_age"]

                # 
                if lower_age_of_ref not in [None, ""]:
                    if is_float( lower_age_of_ref ):
                        lower_age_of_ref = float( lower_age_of_ref )

                if upper_age_of_ref not in [None, ""]:
                    if is_float( upper_age_of_ref ):
                        upper_age_of_ref = float( upper_age_of_ref )


                # 如果需要考虑性别
                # 患者的性别是否与该参考范围相符合
                match_of_gender = False

                if gender_code_of_ref != "":
                    # 
                    if gender_code_of_ref == gender_code_of_patient:
                        match_of_gender = True
                # 如果不需要考虑性别，match_of_gender 直接设置为 True 即可
                else:
                    match_of_gender = True

                # 如果需要考虑年龄范围
                match_of_age = False

                if age_unit_of_ref != "":
                    # 意味着参考范围有年龄使用范围, 目前仅考虑 周岁
                    if age_unit_of_ref == '岁' and type(lower_age_of_ref) == float and type(upper_age_of_ref) == float:
                        # 观察患者的年龄是否落在参考范围内
                        if age_num_of_patient >= lower_age_of_ref and age_num_of_patient <= upper_age_of_ref:
                            match_of_age = True
                # 如果不需要考虑年龄范围, 年龄适配直接设置为True
                else:
                    match_of_age = True

                # 适用改组参考范围，使用该组参考范围
                if match_of_gender and match_of_age:
                    used_ref_range = tmpinfo
                    selected_proper_range = True
                    break

            # 如果没有选择到适合的参考范围，别随意使用参考范围
            if selected_proper_range == False:
                pass


    #
    return used_ref_range


# 属性词识别之前，需要结合 core_words_found 对 text_str 进行 mask 处理
def get_masked_text_str( text_str, keywords_found ):
    # 
    masked_text_str = text_str

    # 对 masked_text_str ，基于 keywords_found 进行精准的更改
    for keyword_found in keywords_found:
        span_info, span_start, span_end = keyword_found
        #
        left_context = masked_text_str[0:span_start]
        right_context = masked_text_str[span_end+1:]
        #
        # span_str = masked_text_str[span_start:span_end+1]
        span_str = masked_text_str[span_start:span_end+1]
        mask_str = '$'*len(span_str)
        # update masked_text
        masked_text_str = left_context + mask_str + right_context    

    #
    return masked_text_str


# 对扫描到的核心词进行清洗，目前加入的主要清洗规则是 
def filter_keywords_found( text_str, keywords_found ):
    #
    cleaned_keywords_found = []

    #
    for keyword_found in keywords_found:
        span_info, span_start, span_end = keyword_found    

        # 先看这个字符串是不是纯粹由字母或数字构成
        # 利用 python 的 isalnum()函数, 这样简单点
        span_str = text_str[span_start:span_end+1]
        # print( " span_str ", span_str.isalnum(), span_str ) 

        # 如果该实体纯粹由 字母和数字构成，需要判断是否保留
        # 默认保留该实体
        keep_it_flag = True

        # S.isalpha() :可判断字符串S是否全为字母，但判断中文时依旧会返回True
        # 解决方法：先将字符串s进行编码
        if span_str.encode().isalnum():
            # 前一个字符如果是数字或字母
            if span_start - 1 >= 0:
                if text_str[span_start - 1].encode().isalnum():
                    keep_it_flag = False
            # 后一个字符如果是字母
            # WBC6.21被清洗掉了
            if span_end+1 <= len(text_str)-1:
                if text_str[span_end+1].encode().isalpha():
                    keep_it_flag = False
        
        # 去除属性词中的屏蔽词
        if re.search('屏蔽词', span_info):
            keep_it_flag = False

        # 
        if keep_it_flag:
            cleaned_keywords_found.append( keyword_found )

    #
    return cleaned_keywords_found



# 求解核心词的作用范围
def get_context_boundary_of_core_word( text_str, core_words_found, core_word_idx ):
    # 求解目标
    # boundary_start, boundary_end

    # 提取目标核心词的信息
    core_word_info = core_words_found[core_word_idx]

    # 提取目标核心词的信息
    span_info, span_start, span_end = core_word_info


    # 初始化 +-3
    boundary_start = max(span_start - 3, 0)
    boundary_end   = min(span_end   + 3, len(text_str)-1 )


    # [MEMO] 先General, 未来可细化
    # 细化1: 不同类别核心词的SCOPE设置
    # 细化2: 左右作用范围的不对称设置

    # 表型核心词、部位核心词 的作用范围
    # a. 定位 ent_span_start 之前的第一个逗号、分号或句号 在文本中所在的位置 
    # b. 定位 ent_span_end 之后的第一个逗号、分号或句号 在文本中所在的位置 
    # 双侧肺部     严重头晕、头痛
    if re.search("表型核心词|部位核心词", span_info):
        # initialize
        # 作用范围最大边界 cutoff，之前是15
        shift_cutoff = 12
        # ent_span_start - 15 或 文本开头
        boundary_start = max(span_start - shift_cutoff, 0)
        # ent_span_end + 15    或文本结束
        boundary_end   = min(span_end   + shift_cutoff, len(text_str)-1 )

        # a. 定位 ent_span_start 之前的第一个逗号、分号或句号 在文本中所在的位置 
        shift_count = 0

        for char in text_str[0:span_start][::-1]:
            # 偏移计数
            shift_count += 1
            # 如果遇到下述符号, 即是边界
            # bug 修复 扫描到标点符号后没有breaK
            if char in ',，;；。\n和': 
                # 更新 boundary_start 位置
                boundary_start = span_start - shift_count
                break

        # b. 定位 ent_span_end 之后的第一个逗号、分号或句号 在文本中所在的位置 
        shift_count = 0
        for char in text_str[span_end+1:]:
            # 偏移计数
            shift_count += 1
            # 如果遇到下述符号, 即是边界
            if char in ',，;；。\n和':        
                # 更新 boundary_end 位置
                boundary_end = span_end + shift_count
                break            

    # 确定 SCOPE[0] 和 SCOPE[1] 在 attr_words_found 中的位置, 截取SCOPE区间内的属性词，与核心词进行关联
    # [('轻度||严重程度||表型核心词', 3, 4)] 

    # 测量核心词
    # WBC、RBC均正常
    # a. boundary_start 测量核心词之后的+1的位置
    # b. boundary_end   测量核心词之后的标点符号位置
    if re.search("测量核心词", span_info):
        # initialize

        # a. boundary_start span_end + 1
        boundary_start = min(span_end + 1, len(text_str)-1)


        # b. 定位 ent_span_end 之后的第一个逗号、分号或句号 在文本中所在的位置 
        # 作用范围最大边界 cutoff
        shift_cutoff = 15  
        # ent_span_end + 15    或文本结束
        boundary_end   = min(span_end + shift_cutoff, len(text_str)-1 )

        #
        shift_count = 0
        for char in text_str[span_end+1:]:
            # 偏移计数
            shift_count += 1
            # 如果遇到下述符号, 即是边界 [增加、顿号]
            # if char in ',，;；。、\n和':
            if char in ',，;；。\n和':
                # 更新 boundary_end 位置
                boundary_end = span_end + shift_count
                break              


    # 药物核心词
    # 口服青霉素500mg,每日4次，甲硝唑400mg,每日3次
    # 左氧氟沙星0.5g,每日1次,联合甲硝唑每次0.2g,每日3次,疗程4天
    # 如硫酸亚铁0.3g,每日3次;或右旋糖酐铁50mg,每日2~3次
    # a. boundary_start 药物核心词之后的+1的位置
    # b. boundary_end   
    # 药物核心词之后的标点符号位置(不考虑逗号)
    # 药物核心之后的另一个核心词位置
    # 药物核心词之后的最大作用范围 (+25)
    # 盐酸氨溴索,30mg,每日3次
    if re.search("药物核心词", span_info):
        # initialize

        # a. boundary_start span_end + 1
        boundary_start = min(span_end + 1, len(text_str)-1)


        # b. 定位 ent_span_end 之后的第一个分号或句号 在文本中所在的位置 
        # 作用范围最大边界 cutoff
        shift_cutoff = 30  
        # ent_span_end + 30    或文本结束,
        boundary_end   = min(span_end + shift_cutoff, len(text_str)-1 )

        # 是否存在下一个核心词，如果存在，且比 上述 boundary_end 小，亦可作为天然边界
        if core_word_idx+1  <= len(core_words_found) -1:
            #
            next_core_word_info = core_words_found[core_word_idx+1]
            #
            next_core_span_start = next_core_word_info[1]
            # 
            boundary_end = min( boundary_end, next_core_span_start )

        # 分号或句号 也可作为边界，如果比核心词的位置还小，将之作为边界
        shift_count = 0
        for char in text_str[span_end+1:]:
            # 偏移计数
            shift_count += 1
            # 如果遇到下述符号, 即是边界
            if char in ';；。\n和':        
                # 更新 boundary_end 位置
                boundary_end = min(span_end + shift_count, boundary_end)
                break       

    #
    return (boundary_start, boundary_end)



# 搜索 药物核心词 的 给药剂量属性
def search_ent_info_of_drug_dose( text_str, annotated_ent_info, ent_id, sorted_list_of_ent_ids ):
    # 求解目标
    ent_info_of_drug_dose = {}


    # [前置信息][数据准备]
    # 剂量单位
    dict_of_dose_units = {}
    # 固体、半固体剂型药物常用计量单位
    dict_of_dose_units.setdefault( "mg", ["毫克"])
    dict_of_dose_units.setdefault( "g",  ["克"])
    dict_of_dose_units.setdefault( "μg",  ["微克","ug"])
    # 丸剂、散剂、片剂
    dict_of_dose_units.setdefault( "片",  ["tablet"])
    dict_of_dose_units.setdefault( "包",  [])
    dict_of_dose_units.setdefault( "袋",  [])
    dict_of_dose_units.setdefault( "丸",  [])
    dict_of_dose_units.setdefault( "粒",  [])
    # 饮剂
    dict_of_dose_units.setdefault( "剂",  [])
    dict_of_dose_units.setdefault( "副",  [])
    # 液体剂型药物的常用计量单位
    dict_of_dose_units.setdefault( "L", ["升"])
    dict_of_dose_units.setdefault( "ml",  ["毫升"])
    # 一些抗生素、激素、维生素等药物
    dict_of_dose_units.setdefault( "IU", ["国际单位"])
    dict_of_dose_units.setdefault( "U",  ["单位"])


    # 剂量单位的 pattern 和 剂量单位的标准化
    dose_units = set()
    mapping_of_dose_units = {}

    for std_unit in dict_of_dose_units:
        syn_units = dict_of_dose_units[std_unit]
        #
        dose_units.add( std_unit )
        mapping_of_dose_units.setdefault( std_unit, std_unit )

        for syn_unit in syn_units:
            dose_units.add( syn_unit )
            mapping_of_dose_units.setdefault( syn_unit, std_unit )

    # 剂量单位的 pattern
    dose_units = list( dose_units )

    # 不需要 re.escape
    pattern_of_dose_units = '|'.join( dose_units )   



    # 核心词 信息提取
    # 实体信息 
    ent_info = annotated_ent_info[ent_id]

    # 核心词的位置
    span_start_of_drug_ent = ent_info['span_start']
    span_end_of_drug_ent   = ent_info['span_end']    


    # 是否是 drug_ent
    # 用于占位
    is_drug_ent = True


    if is_drug_ent:
        # 确定 药物核心词的 right context boundary
        # 参考 get_context_boundary_of_core_word() 函数

        # a. boundary_start span_end + 1, len(text_str)-1 min
        boundary_start = min(span_end_of_drug_ent + 1, len(text_str)-1)

        # b. 定位 ent_span_end 之后的第一个分号或句号 在文本中所在的位置 
        # 作用范围最大边界 cutoff
        shift_cutoff = 30  
        # ent_span_end + 30    或文本结束,
        boundary_end   = min(span_end_of_drug_ent + shift_cutoff, len(text_str)-1 ) 

        # 是否存在下一个核心词，如果存在，且比 上述 boundary_end 小，亦可作为天然边界
        tmp_item = ( span_start_of_drug_ent, ent_id )

        if tmp_item in sorted_list_of_ent_ids:
            item_idx = sorted_list_of_ent_ids.index( tmp_item )
            # 
            if item_idx+1 <= len(sorted_list_of_ent_ids) -1:
                next_item = sorted_list_of_ent_ids[ item_idx+1 ]
                # 
                next_item_span_start = next_item[0]
                #
                boundary_end = min( boundary_end, next_item_span_start )

        # 分号或句号 也可作为边界，如果比核心词的位置还小，将之作为边界
        shift_count = 0
        for char in text_str[span_end_of_drug_ent+1:]:
            # 偏移计数
            shift_count += 1
            # 如果遇到下述符号, 即是边界
            if char in ';；。\n和':        
                # 更新 boundary_end 位置
                boundary_end = min(span_end_of_drug_ent + shift_count, boundary_end)
                break                   


        # 在 right context 中搜索 给药剂量 属性实体
        # 如果能有效关联，对 给药剂量 表达模式 标准化  

        # 首先搜索 剂量单位           
        # 检索 剂量单位 时忽略大小写
        exist_dose_unit = True

        # 如果能搜索到 剂量单位 ，记录 剂量单位 的以下信息
        str_of_dose_unit = ""
        span_start_of_dose_unit = ""
        span_end_of_dose_unit   = ""            

        #
        # print( "**** ", text_str[boundary_start: boundary_end+1] )
        # 搜索 剂量单位 (在药物核心词的 boundary_start 和 boundary_end 中)
        # max boundary_end len(text_str)-1
        match_object_of_dose_unit = re.search( pattern_of_dose_units, 
                                                text_str[boundary_start: boundary_end+1], re.I )


        # 如果能搜索到 剂量单位
        if match_object_of_dose_unit:
            # 测量单位在文中的位置坐标
            span_start_of_dose_unit = match_object_of_dose_unit.start() + boundary_start
            span_end_of_dose_unit   = match_object_of_dose_unit.end()   + boundary_start
            #
            str_of_dose_unit = match_object_of_dose_unit.group()  
            # print( "***** ", str_of_dose_unit)
        # 如果搜索不到 剂量单位
        else:
            # print("False ")
            exist_dose_unit = False


        # 在能找到 剂量单位 的前提下，才触发下列操作
        # 剂量数值
        # 如果 exist_dose_unit = True
        # 查看 核心词 和 剂量单位 之间是否存在合法的value
        exist_dose_value = False

        str_of_dose_value = ""
        span_start_of_dose_value = ""
        span_end_of_dose_value   = ""   


        # 在能找到 剂量单位 的前提下，继续搜索剂量数值
        if exist_dose_unit:
            #
            # float \d*\.?\d+
            # float -~ float  \d*\.?\d+.{1,3}\d*\.?\d+.
            pattern_of_dose_values = ' \d*\.?\d+.{1,3}\d*\.?\d+.|\d*\.?\d+'


            if exist_dose_unit:                        
                # 
                # 搜索 核心词 和 剂量单位之间 的数值模式
                match_object_of_dose_value = re.search( pattern_of_dose_values, 
                                                      text_str[boundary_start: span_start_of_dose_unit] )  

                #
                # 如果可以搜索到 match_object_of_value
                if match_object_of_dose_value:
                    # 剂量数值 在文本中的坐标
                    span_start_of_dose_value = match_object_of_dose_value.start() + boundary_start
                    span_end_of_dose_value   = match_object_of_dose_value.end()   + boundary_start
                    str_of_dose_value = match_object_of_dose_value.group() 

                    # 剂量数值和核心词之间的距离不能过远
                    # 如果过远，则认为是虚假的 剂量数值                 
                    if abs(span_start_of_dose_value - span_end_of_drug_ent) <= 25:
                        exist_dose_value = True


        # 
        # 同时存在 剂量单位 和 剂量复制，才触发下列操作
        # 对于 该药物实体，若存在符合规则 的 剂量数值 和 剂量单位
        # 剂量数值 和 剂量单位 的标准化
        std_dose_value = str_of_dose_value
        std_dose_unit  = str_of_dose_unit          

        # 同时存在 剂量单位 和 剂量复制，才触发下列操作
        if exist_dose_value and std_dose_unit:
            # 给药剂量 的 ent_info 
            # 求解目标
            ent_info_of_drug_dose = {}

            # dose_unit 的标准化
            if str_of_dose_unit in mapping_of_dose_units:
                std_dose_unit = mapping_of_dose_units[str_of_dose_unit]

            # dose_value 的标准化 
            # 区间型 dose_value 的处理
            if str_of_dose_value.replace('.','',1).isdigit():
                pass
            else:
                # 提取 两个 float 
                list_of_float_values = re.findall( "\d*\.?\d+", str_of_dose_value)

                # 
                if len( list_of_float_values ) == 2:
                    float_value_1 = list_of_float_values[0]
                    float_value_2 = list_of_float_values[1]
                    # 标准化生成 std_dose_value
                    std_dose_value = float_value_1 + ' - ' + float_value_2

            # 准备并记录需要更新的信息
            # 
            span_start_of_drug_dose  = span_start_of_dose_value
            span_end_of_drug_dose    = span_end_of_dose_unit -1
            ent_id_of_dose_value     = str(span_start_of_drug_dose) + ':' + str(span_end_of_drug_dose)
            #
            std_value_of_dose_value  = str(std_dose_value) + ' ' + std_dose_unit

            #
            ent_info_of_drug_dose.setdefault( "ent_id", ent_id_of_dose_value )
            ent_info_of_drug_dose.setdefault( "span_start", span_start_of_drug_dose)
            ent_info_of_drug_dose.setdefault( "span_end",   span_end_of_drug_dose)
            ent_info_of_drug_dose.setdefault( "term_type", "属性词" )
            ent_info_of_drug_dose.setdefault( "term_str",  text_str[span_start_of_drug_dose:span_end_of_drug_dose+1])
            ent_info_of_drug_dose.setdefault( "std_value",  std_value_of_dose_value)
            ent_info_of_drug_dose.setdefault( "attr_name", "每次给药剂量" )
            ent_info_of_drug_dose.setdefault( "attr_of", "药物核心词" )     

    #
    return ent_info_of_drug_dose



# 搜索表型核心词的持续时间属性
# [Add] 后面不能有"前"
def search_ent_info_of_time_duration(text_str, annotated_ent_info, ent_id):
    # 求解目标
    ent_info_of_time_duration = {}


    # [前置信息][数据准备]
    # 时间单位
    # clean_name
    dict_of_time_units = {}
    dict_of_time_units.setdefault('年', ['year', 'y'])
    dict_of_time_units.setdefault('月', ['month','个月'])
    dict_of_time_units.setdefault('周', ['week'])
    dict_of_time_units.setdefault('天', ['day', 'd'])
    dict_of_time_units.setdefault('小时', ['hour', 'h'])
    dict_of_time_units.setdefault('分钟', ['minute'])
    dict_of_time_units.setdefault('秒', ['second', 's'])

    # 时间单位的集合
    allowed_units = set()
    # 时间单位的标准化
    mapping_of_time_units = {}    

    for std_unit in dict_of_time_units:
        syn_units = dict_of_time_units[std_unit]
        #
        allowed_units.add( std_unit )
        mapping_of_time_units.setdefault( std_unit, std_unit )

        for syn_unit in syn_units:
            allowed_units.add( syn_unit )
            mapping_of_time_units.setdefault( syn_unit, std_unit )

    #
    allowed_units = list( allowed_units )

    # 不需要 re.escape
    pattern_of_units = '|'.join( allowed_units )    


    # 时间数值模式 和 时间数值的标准化
    # 中文时间数值 --> 阿拉伯时间数值
    # float
    pattern_of_float_numbers = '\d*\.?\d+'

    # 列表中位置 "零" : 0
    cn_number_list = ['零','一','二','三','四','五','六','七','八','九',"十",'两','半']

    # 
    pattern_of_cn_numbers = '[' + ''.join(cn_number_list) + ']'

    # 0-9 或 零到九
    pattern_of_values = pattern_of_float_numbers + '|' + pattern_of_cn_numbers

    # 
    mapping_of_cn_numbers = {}

    for idx, cn_num in enumerate( cn_number_list ):
        if cn_num not in ['两','半']:
            mapping_of_cn_numbers.setdefault( cn_num, str(idx) )
        elif cn_num == '两':
            mapping_of_cn_numbers.setdefault( cn_num, '2' )
        elif cn_num == '半':
            mapping_of_cn_numbers.setdefault( cn_num, '0.5' )



    # 信息提取
    # 实体信息 
    ent_info = annotated_ent_info[ent_id]

    # 核心词的位置
    span_start_of_phenotype_ent = ent_info['span_start']
    span_end_of_phenotype_ent   = ent_info['span_end']


    # 是否是 phenotype_ent
    # 占位
    is_phenotype_ent = True    


    if is_phenotype_ent:
        # 尝试寻找 持续时间 实体
        # float + unit 支持 1~2 个空格或其它字符(余)
        # 10年  10 年 10余年 1.5年
        # 像识别 测量结果 实体一样 识别 持续时间实体
        # 先识别 时间单位
        # 再识别 表型实体 和 时间单位 中间的 数字
        # 考虑到 数字可能是汉字 形式，可先 cn2an

        # 搜索核心词的一定范围内的right_context 是否存在 allowed_units
        # 检索时间单位时忽略大小写
        exist_allow_unit = True

        # 如果能搜索到测量单位，记录测量单位的以下信息
        str_of_unit = ""
        span_start_of_measure_unit = ""
        span_end_of_measure_unit   = ""

        # 搜索
        # 边界处理
        boundary_start = span_start_of_phenotype_ent
        boundary_end   = min( span_end_of_phenotype_ent+15, len(text_str)-1 )

        # 当 boundary_end = len(text_str)-1 时
        match_object_of_unit = re.search( pattern_of_units, text_str[boundary_start: boundary_end+1], re.I )

        # 如果能搜索到 时间单位
        if match_object_of_unit:
            # 测量单位在文中的位置坐标
            span_start_of_measure_unit = match_object_of_unit.start() + boundary_start
            span_end_of_measure_unit   = match_object_of_unit.end()-1   + boundary_start
            #
            str_of_unit = match_object_of_unit.group() 

            # print( "info ", span_start_of_measure_unit, span_end_of_measure_unit, str_of_unit )

            # 如果 测量核心词 和 测量单位 之间有以下符号
            # 那么 exist_allow_unit 设置为 False
            for char in text_str[span_end_of_phenotype_ent: span_start_of_measure_unit]:
                if char in ',，;；。\n':
                    exist_allow_unit = False  

            # 如果 测量单位之后有"前", 说明描述的不是持续时间
            if span_end_of_measure_unit+1 <= len(text_str) - 1:
                if text_str[span_end_of_measure_unit+1] == '前':
                    exist_allow_unit = False

        # 如果搜索不到测量单位
        else:
            exist_allow_unit = False            



        # 时间数值
        # 如果 exist_allow_unit = True
        # 查看 核心词和测量单位之间是否存在合法的value
        exist_allow_value = False

        str_of_value = ""
        span_start_of_measure_value = ""
        span_end_of_measure_value   = ""


        # 在存在合法的测量单位时，才触发下列操作
        if exist_allow_unit:
            # 
            # 搜索 核心词 和 单位之间 的数值模式
            match_object_of_value = re.search( pattern_of_values, 
                                                  text_str[span_end_of_phenotype_ent+1: span_start_of_measure_unit] )  

            #
            # 如果可以搜索到 match_object_of_value
            if match_object_of_value:
                # 测量数值在文本中的坐标
                span_start_of_measure_value = match_object_of_value.start() + span_end_of_phenotype_ent+1
                span_end_of_measure_value   = match_object_of_value.end()   + span_end_of_phenotype_ent 
                str_of_value = match_object_of_value.group() 

                # print( "info ", span_start_of_measure_value, span_end_of_measure_value, str_of_value )

                # 测量数值和核心词之间的距离不能过远
                # 如果过远，则认为是虚假的 测量数值                 
                if abs(span_start_of_measure_value - span_end_of_phenotype_ent) <= 5:
                    exist_allow_value = True


        # 对于 该表型实体，若存在符合规则 的 时间数值 和 时间单位
        # 时间数值 和 时间单位 的标准化
        # str_of_value
        # str_of_unit
        std_time_value = str_of_value
        std_time_unit  = str_of_unit


        # 同时存在 符合规则 的 时间数值 和 时间单位，才触发下列操作
        if exist_allow_value and exist_allow_unit:
            # 持续时间的 ent_info 
            ent_info_of_time_duration = {}

            # time_unit 的标准化
            if str_of_unit in mapping_of_time_units:
                std_time_unit = mapping_of_time_units[str_of_unit]
            # time_value 的标准化 (针对中文数字)
            if str_of_value in mapping_of_cn_numbers:
                std_time_value = mapping_of_cn_numbers[str_of_value]

            # 准备并记录需要更新的信息
            # 
            span_start_of_time_duration = span_start_of_measure_value
            span_end_of_time_duration   = span_end_of_measure_unit 
            ent_id_of_time_duration = str(span_start_of_time_duration) + ':' + str(span_end_of_time_duration)
            #
            std_value_of_time_duration = str(std_time_value) + ' ' + std_time_unit

            #
            ent_info_of_time_duration.setdefault( "ent_id", ent_id_of_time_duration )
            ent_info_of_time_duration.setdefault( "span_start", span_start_of_time_duration)
            ent_info_of_time_duration.setdefault( "span_end",   span_end_of_time_duration)
            ent_info_of_time_duration.setdefault( "term_type", "属性词" )
            ent_info_of_time_duration.setdefault( "term_str",  text_str[span_start_of_time_duration:span_end_of_time_duration+1])
            ent_info_of_time_duration.setdefault( "std_value",  std_value_of_time_duration)
            ent_info_of_time_duration.setdefault( "attr_name", "持续时间" )
            ent_info_of_time_duration.setdefault( "attr_of", "表型核心词" ) 

    return ent_info_of_time_duration


# 搜索表型核心词的 既往存在 属性
# 术语型 的既往存在触发词  曾 史 不用在此考虑
# 主要考虑 x年前, x月前
def search_ent_info_of_past_presence(text_str, annotated_ent_info, ent_id):
    # 求解目标
    ent_info_of_past_presence = {}


    # [前置信息][数据准备]
    # 时间单位
    # 主要考虑 x年前, x月前
    # clean_name
    dict_of_time_units = {}
    dict_of_time_units.setdefault('年', ['year', 'y'])
    dict_of_time_units.setdefault('月', ['month','个月'])


    # 既往 时间单位的集合
    # 年前 月前
    allowed_units = set()
    # 既往 时间单位的标准化
    mapping_of_time_units = {}    

    for std_unit in dict_of_time_units:
        syn_units = dict_of_time_units[std_unit]
        #
        allowed_units.add( std_unit +'前' )
        mapping_of_time_units.setdefault( std_unit+'前', std_unit+'前' )

        for syn_unit in syn_units:
            allowed_units.add( syn_unit +'前' )
            mapping_of_time_units.setdefault( syn_unit +'前', std_unit +'前' )

    #
    allowed_units = list( allowed_units )

    # 不需要 re.escape
    pattern_of_units = '|'.join( allowed_units )    


    # 时间数值模式 和 时间数值的标准化
    # 中文时间数值 --> 阿拉伯时间数值
    # float
    pattern_of_float_numbers = '\d*\.?\d+'

    # 列表中位置 "零" : 0
    cn_number_list = ['零','一','二','三','四','五','六','七','八','九',"十",'两','半']

    # 
    pattern_of_cn_numbers = '[' + ''.join(cn_number_list) + ']'

    # 0-9 或 零到九
    pattern_of_values = pattern_of_float_numbers + '|' + pattern_of_cn_numbers

    # 
    mapping_of_cn_numbers = {}

    for idx, cn_num in enumerate( cn_number_list ):
        if cn_num not in ['两','半']:
            mapping_of_cn_numbers.setdefault( cn_num, str(idx) )
        elif cn_num == '两':
            mapping_of_cn_numbers.setdefault( cn_num, '2' )
        elif cn_num == '半':
            mapping_of_cn_numbers.setdefault( cn_num, '0.5' )



    # 信息提取
    # 实体信息 
    ent_info = annotated_ent_info[ent_id]

    # 核心词的位置
    span_start_of_phenotype_ent = ent_info['span_start']
    span_end_of_phenotype_ent   = ent_info['span_end']


    # 是否是 phenotype_ent
    # 占位
    is_phenotype_ent = True    


    if is_phenotype_ent:
        # 尝试寻找 持续时间 实体
        # float + unit 支持 1~2 个空格或其它字符(余)
        # 10年  10 年 10余年 1.5年
        # 像识别 测量结果 实体一样 识别 持续时间实体
        # 先识别 时间单位
        # 再识别 表型实体 和 时间单位 中间的 数字
        # 考虑到 数字可能是汉字 形式，可先 cn2an

        # 搜索核心词的一定范围内的right_context 是否存在 allowed_units
        # 检索时间单位时忽略大小写
        exist_allow_unit = True

        # 如果能搜索到测量单位，记录测量单位的以下信息
        str_of_unit = ""
        span_start_of_measure_unit = ""
        span_end_of_measure_unit   = ""

        # 搜索 (往前搜索)
        # 边界限制
        boundary_start = max( span_start_of_phenotype_ent-15, 0 )
        boundary_end   = min( span_end_of_phenotype_ent-1, len(text_str)-1 )
        # 
        match_object_of_unit = re.search( pattern_of_units, text_str[boundary_start: boundary_end+1], re.I )

        # 如果能搜索到 时间单位
        if match_object_of_unit:
            # 测量单位在文中的位置坐标
            span_start_of_measure_unit = match_object_of_unit.start() + boundary_start
            span_end_of_measure_unit   = match_object_of_unit.end()-1 + boundary_start
            #
            str_of_unit = match_object_of_unit.group() 
            # print( "\n span ", span_start_of_measure_unit, span_end_of_measure_unit, str_of_unit, match_object_of_unit.span() )

            # 如果 测量核心词 和 测量单位 之间有以下符号
            # 那么 exist_allow_unit 设置为 False
            for char in text_str[span_end_of_measure_unit: span_start_of_phenotype_ent]:
                if char in ',，;；。\n':
                    exist_allow_unit = False  


        # 如果搜索不到测量单位
        else:
            exist_allow_unit = False            



        # 时间数值
        # 如果 exist_allow_unit = True
        # 查看 核心词和测量单位之间是否存在合法的value
        exist_allow_value = False

        str_of_value = ""
        span_start_of_measure_value = ""
        span_end_of_measure_value   = ""


        # 在存在合法的测量单位时，才触发下列操作
        if exist_allow_unit:
            # 
            # 搜索测量单位前的测量数值 x年前
            boundary_start = max( 0, span_start_of_measure_unit -3)
            boundary_end   = span_end_of_measure_unit 

            #
            match_object_of_value = re.search( pattern_of_values, text_str[boundary_start: boundary_end] )  

            #
            # 如果可以搜索到 match_object_of_value
            if match_object_of_value:
                # 测量数值在文本中的坐标
                span_start_of_measure_value = match_object_of_value.start() + boundary_start 
                span_end_of_measure_value   = match_object_of_value.end()   + boundary_start +1
                str_of_value = match_object_of_value.group() 
                # print( "span ", span_start_of_measure_value,  span_end_of_measure_value)


                # 测量单位和核心词之间的距离不能过远
                # 如果过远，则认为是虚假的 测量数值                 
                if abs(span_end_of_measure_unit - span_start_of_phenotype_ent) <= 5:
                    exist_allow_value = True


        # 对于 该表型实体，若存在符合规则 的 时间数值 和 时间单位
        # 时间数值 和 时间单位 的标准化
        # str_of_value
        # str_of_unit
        std_time_value = str_of_value
        std_time_unit  = str_of_unit


        # 同时存在 符合规则 的 时间数值 和 时间单位，才触发下列操作
        if exist_allow_value and exist_allow_unit:
            # 持续时间的 ent_info 
            ent_info_of_past_presence = {}

            # time_unit 的标准化
            if str_of_unit in mapping_of_time_units:
                std_time_unit = mapping_of_time_units[str_of_unit]
            # time_value 的标准化 (针对中文数字)
            if str_of_value in mapping_of_cn_numbers:
                std_time_value = mapping_of_cn_numbers[str_of_value]

            # 准备并记录需要更新的信息
            # 
            span_start_of_past_presence = span_start_of_measure_value
            span_end_of_past_presence   = span_end_of_measure_unit 
            ent_id_of_past_presence = str(span_start_of_past_presence) + ':' + str(span_end_of_past_presence)
            # 既往存在
            std_value_of_past_presence = "既往存在"

            #
            ent_info_of_past_presence.setdefault( "ent_id", ent_id_of_past_presence )
            ent_info_of_past_presence.setdefault( "span_start", span_start_of_past_presence)
            ent_info_of_past_presence.setdefault( "span_end",   span_end_of_past_presence)
            ent_info_of_past_presence.setdefault( "term_type", "属性词" )
            ent_info_of_past_presence.setdefault( "term_str",  text_str[span_start_of_past_presence:span_end_of_past_presence+1])
            ent_info_of_past_presence.setdefault( "std_value",  std_value_of_past_presence)
            ent_info_of_past_presence.setdefault( "attr_name", "存在情况" )
            ent_info_of_past_presence.setdefault( "attr_of", "表型核心词" ) 

    return ent_info_of_past_presence



# 搜索 药物核心词 的 给药频率 属性
def search_ent_info_of_drug_freq( text_str, annotated_ent_info, ent_id, sorted_list_of_ent_ids ):
    # 求解目标
    ent_info_of_drug_freq = {}

    # 每日三次
    # 一天三次
    # 每12小时一次


    # [前置信息][数据准备]


    #
    # 时间频率 术语的标准化
    mapping_of_time_expressions = {}
    # ['每日', '一天', '每天'] --> "每天"
    mapping_of_time_expressions.setdefault('每天', '每天')
    mapping_of_time_expressions.setdefault('每日', '每天')
    mapping_of_time_expressions.setdefault('一天', '每天')
    mapping_of_time_expressions.setdefault('一日', '每天')


    # 中文数字
    cn_number_list = ['零','一','二','三','四','五','六','七','八','九',"十",'两','半']

    # 中文数字搜索模式
    pattern_of_cn_numbers = '[' + ''.join(cn_number_list) + ']'


    # 中文数字到阿拉伯数字的映射
    mapping_of_cn_numbers = {}

    for idx, cn_num in enumerate( cn_number_list ):
        if cn_num not in ['两','半']:
            mapping_of_cn_numbers.setdefault( cn_num, str(idx) )
        elif cn_num == '两':
            mapping_of_cn_numbers.setdefault( cn_num, '2' )
        elif cn_num == '半':
            mapping_of_cn_numbers.setdefault( cn_num, '0.5' )





    # 核心词 信息提取
    # 实体信息 
    ent_info = annotated_ent_info[ent_id]

    # 核心词的位置
    span_start_of_drug_ent = ent_info['span_start']
    span_end_of_drug_ent   = ent_info['span_end']    


    # 是否是 drug_ent
    # 用于占位
    is_drug_ent = True


    if is_drug_ent:
        # 确定 药物核心词的 right context boundary
        # 参考 get_context_boundary_of_core_word() 函数

        # a. boundary_start span_end + 1, len(text_str)-1 min
        boundary_start = min(span_end_of_drug_ent + 1, len(text_str)-1)

        # b. 定位 ent_span_end 之后的第一个分号或句号 在文本中所在的位置 
        # 作用范围最大边界 cutoff
        shift_cutoff = 30  
        # ent_span_end + 30    或文本结束,
        boundary_end   = min(span_end_of_drug_ent + shift_cutoff, len(text_str)-1 ) 

        # 是否存在下一个核心词，如果存在，且比 上述 boundary_end 小，亦可作为天然边界
        tmp_item = ( span_start_of_drug_ent, ent_id )

        if tmp_item in sorted_list_of_ent_ids:
            item_idx = sorted_list_of_ent_ids.index( tmp_item )
            # 
            if item_idx+1 <= len(sorted_list_of_ent_ids) -1:
                next_item = sorted_list_of_ent_ids[ item_idx+1 ]
                # 
                next_item_span_start = next_item[0]
                #
                boundary_end = min( boundary_end, next_item_span_start )

        # 分号或句号 也可作为边界，如果比核心词的位置还小，将之作为边界
        shift_count = 0
        for char in text_str[span_end_of_drug_ent+1:]:
            # 偏移计数
            shift_count += 1
            # 如果遇到下述符号, 即是边界
            if char in ';；。\n和':        
                # 更新 boundary_end 位置
                boundary_end = min(span_end_of_drug_ent + shift_count, boundary_end)
                break                   


        # 在 right context 中搜索 给药频率 属性实体
        # 如果能有效关联，对 给药频率 表达模式 标准化

        # 
        # 每天三次
        pattern_of_dose_freq_1 = '[每一][天日][\d一二三四五六七八九两]次'
        # 每2小时1次 每十二小时
        pattern_of_dose_freq_2 = '每[\d一二三四五六七八九两]+小时[\d一二三四五六七八九两]次'

        #
        pattern_of_dose_freq = '|'.join( [pattern_of_dose_freq_1, pattern_of_dose_freq_2] )


        # 首先搜索 给药频次           
        exist_dose_freq = True

        # 如果能搜索到 给药频次 ，记录 给药频次 的以下信息
        str_of_dose_freq = ""
        span_start_of_dose_freq = ""
        span_end_of_dose_freq   = ""            

        #
        # 搜索 给药频次 (在药物核心词的 boundary_start 和 boundary_end 中)
        # max boundary_end len(text_str)-1
        match_object_of_dose_freq = re.search( pattern_of_dose_freq, 
                                                text_str[boundary_start: boundary_end+1], re.I )


        # 如果能搜索到 给药频次
        if match_object_of_dose_freq:
            # 测量单位在文中的位置坐标
            span_start_of_dose_freq = match_object_of_dose_freq.start() + boundary_start
            span_end_of_dose_freq   = match_object_of_dose_freq.end()   + boundary_start
            #
            str_of_dose_freq = match_object_of_dose_freq.group()  
        # 如果搜索不到 给药频次
        else:
            exist_dose_freq = False


        # 在能找到 给药频次 的前提下，才触发下列操作
        # 给药频次 的 标准化 处理
        std_dose_freq = str_of_dose_freq

        # 存在 给药频次 ，才触发下列操作
        if exist_dose_freq :
            # 给药剂量 的 ent_info 
            # 求解目标
            ent_info_of_drug_freq = {}

            # 每日 的标准化 --> 每日       
            std_time_expression = ""

            match_object_of_time_term = re.search( "[每一][天日]", str_of_dose_freq)
            if match_object_of_time_term:
                real_time_expression = match_object_of_time_term.group()
                #
                if real_time_expression in mapping_of_time_expressions:
                    std_time_expression = mapping_of_time_expressions[real_time_expression]

            # 每两小时的标准化  每 2 小时
            match_object_of_time_term = re.search( "每[\d一二三四五六七八九两]+小时", str_of_dose_freq)
            if match_object_of_time_term:
                std_time_expression = match_object_of_time_term.group()

                # 如果存在 cn_number                
                match_of_cn_count = re.search( '[一二三四五六七八九两]', match_object_of_time_term.group() )
                if match_of_cn_count:
                    cn_count_num = match_of_cn_count.group()
                    if cn_count_num in mapping_of_cn_numbers:
                        std_count_num = mapping_of_cn_numbers[cn_count_num]
                        #
                        std_time_expression = std_time_expression.replace(cn_count_num, std_count_num) 
         

            # 多少次 的标准化
            std_count_expressions = ""

            match_object_of_count_term = re.search( "[\d一二三四五六七八九两]次", str_of_dose_freq)
            if match_object_of_count_term:
                # \d次
                std_count_expressions = match_object_of_count_term.group()

                # 如果存在 cn_number                
                match_of_cn_count = re.search( '[一二三四五六七八九两]', match_object_of_count_term.group() )
                if match_of_cn_count:
                    cn_count_num = match_of_cn_count.group()
                    if cn_count_num in mapping_of_cn_numbers:
                        std_count_num = mapping_of_cn_numbers[cn_count_num]
                        #
                        std_count_expressions = std_count_expressions.replace(cn_count_num, std_count_num) 

             

            # 准备并记录需要更新的信息
            # 
            span_start_of_drug_freq  = span_start_of_dose_freq
            span_end_of_drug_freq    = span_end_of_dose_freq -1
            ent_id_of_dose_freq     = str(span_start_of_drug_freq) + ':' + str(span_end_of_drug_freq)
            #
            std_value_of_dose_freq  = str(std_time_expression) + '' + std_count_expressions

            #
            ent_info_of_drug_freq.setdefault( "ent_id", ent_id_of_dose_freq )
            ent_info_of_drug_freq.setdefault( "span_start", span_start_of_drug_freq)
            ent_info_of_drug_freq.setdefault( "span_end",   span_end_of_drug_freq)
            ent_info_of_drug_freq.setdefault( "term_type", "属性词" )
            ent_info_of_drug_freq.setdefault( "term_str",  text_str[span_start_of_dose_freq:span_end_of_drug_freq+1])
            ent_info_of_drug_freq.setdefault( "std_value",  std_value_of_dose_freq)
            ent_info_of_drug_freq.setdefault( "attr_name", "给药频率" )
            ent_info_of_drug_freq.setdefault( "attr_of", "药物核心词" )     

    #
    return ent_info_of_drug_freq




# 搜索 药物核心词 的 连续给药时间 属性
def search_ent_info_of_drug_duration( text_str, annotated_ent_info, ent_id, sorted_list_of_ent_ids ):
    # 求解目标
    ent_info_of_drug_duration = {}

    # 连续3天
    # 连用5天


    # [前置信息][数据准备]


    #
    # 时间频率 术语的标准化
    mapping_of_continue_expressions = {}
    # ['连续', '连用'] --> "每天"
    mapping_of_continue_expressions.setdefault('连续', '连用')
    mapping_of_continue_expressions.setdefault('连用', '连用')


    # 中文数字
    cn_number_list = ['零','一','二','三','四','五','六','七','八','九',"十",'两','半']

    # 中文数字搜索模式
    pattern_of_cn_numbers = '[' + ''.join(cn_number_list) + ']'


    # 中文数字到阿拉伯数字的映射
    mapping_of_cn_numbers = {}

    for idx, cn_num in enumerate( cn_number_list ):
        if cn_num not in ['两','半']:
            mapping_of_cn_numbers.setdefault( cn_num, str(idx) )
        elif cn_num == '两':
            mapping_of_cn_numbers.setdefault( cn_num, '2' )
        elif cn_num == '半':
            mapping_of_cn_numbers.setdefault( cn_num, '0.5' )


    # 核心词 信息提取
    # 实体信息 
    ent_info = annotated_ent_info[ent_id]

    # 核心词的位置
    span_start_of_drug_ent = ent_info['span_start']
    span_end_of_drug_ent   = ent_info['span_end']    


    # 是否是 drug_ent
    # 用于占位
    is_drug_ent = True


    if is_drug_ent:
        # 确定 药物核心词的 right context boundary
        # 参考 get_context_boundary_of_core_word() 函数

        # a. boundary_start span_end + 1, len(text_str)-1 min
        boundary_start = min(span_end_of_drug_ent + 1, len(text_str)-1)


        # b. 定位 ent_span_end 之后的第一个分号或句号 在文本中所在的位置 
        # 作用范围最大边界 cutoff
        shift_cutoff = 30  
        # ent_span_end + 30    或文本结束,
        boundary_end   = min(span_end_of_drug_ent + shift_cutoff, len(text_str)-1 ) 



        # 是否存在下一个核心词，如果存在，且比 上述 boundary_end 小，亦可作为天然边界
        tmp_item = ( span_start_of_drug_ent, ent_id )

        if tmp_item in sorted_list_of_ent_ids:
            item_idx = sorted_list_of_ent_ids.index( tmp_item )
            # 
            if item_idx+1 <= len(sorted_list_of_ent_ids) -1:
                next_item = sorted_list_of_ent_ids[ item_idx+1 ]
                # 
                next_item_span_start = next_item[0]
                #
                boundary_end = min( boundary_end, next_item_span_start )

        # 分号或句号 也可作为边界，如果比核心词的位置还小，将之作为边界
        shift_count = 0
        for char in text_str[span_end_of_drug_ent+1:]:
            # 偏移计数
            shift_count += 1
            # 如果遇到下述符号, 即是边界
            if char in ';；。\n和':        
                # 更新 boundary_end 位置
                boundary_end = min(span_end_of_drug_ent + shift_count, boundary_end)
                break                   


        # 在 right context 中搜索 给药频率 属性实体
        # 如果能有效关联，对 给药频率 表达模式 标准化

        # 
        # 连用三天
        # 连用三周
        pattern_of_drug_duration = '[连][续用][\d一二三四五六七八九两]+[天周]'



        # 首先搜索 连用时间           
        exist_drug_duration = True

        # 如果能搜索到 连用时间 ，记录 连用时间 的以下信息
        str_of_drug_duration = ""
        span_start_of_drug_duration = ""
        span_end_of_drug_duration   = ""            

        #
        # 搜索 连用时间 (在药物核心词的 boundary_start 和 boundary_end 中)
        # max boundary_end len(text_str)-1
        # print( "here ", text_str[boundary_start: boundary_end+1] )
        match_object_of_drug_duration = re.search( pattern_of_drug_duration, 
                                                text_str[boundary_start: boundary_end+1], re.I )


        # 如果能搜索到 连用时间
        if match_object_of_drug_duration:
            # 测量单位在文中的位置坐标
            span_start_of_drug_duration = match_object_of_drug_duration.start() + boundary_start
            span_end_of_drug_duration   = match_object_of_drug_duration.end()   + boundary_start
            #
            str_of_drug_duration = match_object_of_drug_duration.group()  
        # 如果搜索不到 连用时间
        else:
            exist_drug_duration = False


        # 在能找到 连用时间 的前提下，才触发下列操作
        # 连用时间 的 标准化 处理
        std_drug_duration = str_of_drug_duration

        # 存在 连用时间 ，才触发下列操作
        if exist_drug_duration :
            # 连用时间 的 ent_info 
            # 求解目标
            # ent_info_of_drug_duration = {}

            # 连续 连用
            match_of_continue_term = re.search( '连[续用]', str_of_drug_duration )
            if match_of_continue_term:
                real_continue_term = match_of_continue_term.group()
                if real_continue_term in mapping_of_continue_expressions:
                    std_continue_term = mapping_of_continue_expressions[real_continue_term]
                    #
                    std_drug_duration = str_of_drug_duration.replace(real_continue_term, std_continue_term) 


            # 连用 int 天      
            # 如果存在 cn_number                
            match_of_cn_count = re.search( '[一二三四五六七八九两]', str_of_drug_duration )
            if match_of_cn_count:
                cn_count_num = match_of_cn_count.group()
                if cn_count_num in mapping_of_cn_numbers:
                    std_count_num = mapping_of_cn_numbers[cn_count_num]
                    #
                    std_drug_duration = str_of_drug_duration.replace(cn_count_num, std_count_num) 
         

            # 准备并记录需要更新的信息
            # 
            span_start_of_drug_duration  = span_start_of_drug_duration
            span_end_of_drug_duration    = span_end_of_drug_duration -1
            ent_id_of_drug_duration      = str(span_start_of_drug_duration) + ':' + str(span_end_of_drug_duration)
            #
            std_value_of_drug_duration   = std_drug_duration

            #
            ent_info_of_drug_duration.setdefault( "ent_id", ent_id_of_drug_duration )
            ent_info_of_drug_duration.setdefault( "span_start", span_start_of_drug_duration)
            ent_info_of_drug_duration.setdefault( "span_end",   span_end_of_drug_duration)
            ent_info_of_drug_duration.setdefault( "term_type", "属性词" )
            ent_info_of_drug_duration.setdefault( "term_str",  text_str[span_start_of_drug_duration:span_end_of_drug_duration+1])
            ent_info_of_drug_duration.setdefault( "std_value",  std_value_of_drug_duration)
            ent_info_of_drug_duration.setdefault( "attr_name", "连续给药时间" )
            ent_info_of_drug_duration.setdefault( "attr_of", "药物核心词" )     

    #
    return ent_info_of_drug_duration


