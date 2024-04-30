# -*- coding: utf-8 -*-
# @Date    : 2022-10-12 09:42:15
# @Author  : lizong deng
# @Version : v0.1 (under development)

from suppFunctions import *
import pandas as pd
from flask import Flask, jsonify, request, Response


# define a class named SSUExtractor for extracting ssu from free text
class SSUExtractor(object):
    ###############
    # 类的初始化函数
    ###############
    def __init__(self):
        # 
        # print("Start the initialization of SSUExtractor")

        # 连接数据库所需的相关信息
        self.db_info_dict = {}
        self.db_info_dict.setdefault("host", '192.168.1.224')
        self.db_info_dict.setdefault("port", 3306)
        self.db_info_dict.setdefault("user", 'root')
        self.db_info_dict.setdefault("passwd", '123456')
        self.db_info_dict.setdefault("db", 'phenoalign')
        self.db_info_dict.setdefault("charset", 'utf8')

        # [SSU相关表格]
        # [SSU相关表格 -- 核心词]
        # SSU_TYPE 包含的语义表型
        # "T019": "表型核心词"
        self.ssu_type_of_tags = get_ssu_type_of_tags( self.db_info_dict )
        # print( json.dumps(self.ssu_type_of_tags, ensure_ascii=False, indent=4) )

        # [SSU相关表格 -- 属性词]
        # SSU关联属性表
        # "严重程度": { "SSU_TYPE": "表型核心词", "ATTR_EN_NAME": "Severity", "ATTR_ORDER": "2", "VALUE_SET_TYPE": "TERM", "ALLOW_MULTI_VALUE": "否" },
        self.info_of_ssu_attributes = get_info_of_ssu_attributes(self.db_info_dict)
        # print( json.dumps(self.info_of_ssu_attributes, ensure_ascii=False, indent=4) )

        # 属性关联取值表
        # info_of_std_attributes.setdefault( value_cn_name, value_info_dict)
        # value_cn_name  value_info_dict
        # "不存在": { "VALUE_EN_NAME": "Absent", "SSU_TYPE": "表型核心词", "ATTR_CN_NAME": "存在情况", 
        # "VALUE_ORDER": "2", "VALUE_TRIGGER_CN": null, "VALUE_TRIGGER_EN": null },
        excluded_attributes = ['样本来源']
        included_attributes = 'ALL'
        self.info_of_std_attributes = get_info_of_std_values(self.db_info_dict, excluded_attributes, included_attributes )
        # print( json.dumps(self.info_of_std_attributes, ensure_ascii=False, indent=4) )

        # 针对样本来源单独建立 info_of_sample_names 字典        
        included_attributes = ['样本来源']
        excluded_attributes = 'NONE'
        self.info_of_sample_names = get_info_of_std_values(self.db_info_dict, excluded_attributes, included_attributes )


        # 程序运行时间计算
        # start_time = time.perf_counter()
        # print( "Time used : ", str( round( time.perf_counter() - start_time, 3) ) + ' s' ) 

        # [核心词扫描器]
        # flashtext keyword processor
        # build flashtext keyword processor based on serialized mrconso table 
        coreword_dict_cn = {}

        # 核心词术语来源
        # MRCONSO4CN (contains terms with needed tags)
        # with open("./wenxinkb/mrconso4cn_dict.pkl", 'rb') as ftmp:
            # coreword_dict_cn = pickle.load( ftmp ) 


        # [ADD] 可控的UMLS核心词 (Controlled Vocabulary or Vocabulary under Control)
        # 可控的含义: 经过语料、规则清洗的、能用于标注的、面向应用的 MRCONSO 核心词
        core_words_from_UMLSTermsFiltered = get_corewords_from_UMLSTermsFiltered( self.db_info_dict, self.ssu_type_of_tags )
        # 将这些核心词更新到 coreword_dict_cn
        coreword_dict_cn.update( core_words_from_UMLSTermsFiltered )
        # print case
        case_id = list( core_words_from_UMLSTermsFiltered.keys() ) [0]
        # print( "core_words_from_UMLSTermsFiltered case ", case_id, core_words_from_UMLSTermsFiltered[case_id] )


        # [Supplementary of pub_terms]
        # [ADD] 获取具有 confirmed_cui 的 术语CUI 信息
        # 例如 {"寻常疣": C0043037} 
        # 服务于文本比对函数的补丁 (例如 发热与高热的比对)
        self.dict_of_terms_with_confirmed_cui = get_confirmed_cui_of_terms( self.db_info_dict )
        # print("\nnumbers of terms with confirmed cui ", len(self.dict_of_terms_with_confirmed_cui) )
        # print( "高热", self.dict_of_terms_with_confirmed_cui['高热'] )


        # pub_syns  中文术语的同义词、子结点知识库 [Planet Navigator]
        # 目的 在于服务于 文本比对 函数
        # 只利用了 term_str 信息
        term_knowledge_from_PubSyns = get_term_knowledge_from_PubSyns( self.db_info_dict )

        # 中文术语字符串的 同义词 信息
        # syn_term --> std_term  [pub syns  term_str 形式]
        self.prefered_name_from_PubSyns = get_prefered_name_from_PubSyns( term_knowledge_from_PubSyns )

        # 中文术语字符串的 子结点 信息
        # str_term : [list_of_chidren_terms + list_of_children_cui]
        # cui_term : [list_of_chidren_terms + list_of_children_cui]
        self.children_terms_from_PubSyns = get_children_terms_from_PubSyns( term_knowledge_from_PubSyns, self.dict_of_terms_with_confirmed_cui )

        # 中文术语字符串的 父结点 信息
        # term: []      可以有多个父结点
        # str_term : [list_of_parent_terms + list_of_parent_cui]
        # cui_term : [list_of_parent_terms + list_of_parent_cui]
        self.parent_terms_from_PubSyns = get_parent_terms_from_PubSyns( term_knowledge_from_PubSyns, self.dict_of_terms_with_confirmed_cui )


        # UMLS/SNOMED-CT 的 层级结构知识
        term_knowledge_from_UMLS_Hiers = get_term_knowledge_from_UMLS_Hiers( self.db_info_dict )
        umls_child_cui_to_parent_cuis = term_knowledge_from_UMLS_Hiers[0]
        umls_parent_cui_to_child_cuis = term_knowledge_from_UMLS_Hiers[1]


        # 将来自 umls 的层级结构知识并入到 term_knowledge_from_PubSyns 的 children_terms_from_PubSyns
        # 这种方式可降低 文本比对函数 的代码修改量
        # update self.children_terms_from_PubSyns
        for term_cui in umls_child_cui_to_parent_cuis:
            term_pcuis_from_umls = umls_child_cui_to_parent_cuis[term_cui]
            #
            if term_cui in self.children_terms_from_PubSyns:
                # 并集操作
                combined_parent_terms = self.children_terms_from_PubSyns[term_cui] | term_pcuis_from_umls
                self.children_terms_from_PubSyns[term_cui] = combined_parent_terms
            #
            else:
                self.children_terms_from_PubSyns.setdefault( term_cui, term_pcuis_from_umls )

        # update self.parent_terms_from_PubSyns
        for term_parent_cui in umls_parent_cui_to_child_cuis:
            term_child_cuis_from_umls = umls_parent_cui_to_child_cuis[term_parent_cui]
            #
            if term_parent_cui in self.parent_terms_from_PubSyns:
                # 集合相加
                combined_child_terms = self.parent_terms_from_PubSyns[term_parent_cui] | term_child_cuis_from_umls
                self.parent_terms_from_PubSyns[term_parent_cui] = combined_child_terms
            #
            else:
                self.parent_terms_from_PubSyns.setdefault( term_parent_cui, term_child_cuis_from_umls )        


        # PIAT ： pub_terms
        # 来自 PIAT 核心词知识库的核心词
        core_words_from_PubTerms = get_corewords_from_PubTerms(self.db_info_dict, self.ssu_type_of_tags, self.prefered_name_from_PubSyns)
        # 将这些核心词更新到 coreword_dict_cn
        coreword_dict_cn.update( core_words_from_PubTerms )
        # print case
        case_id = list( core_words_from_PubTerms.keys() ) [0]
        # print( "core_words_from_PubTerms case ", case_id, core_words_from_PubTerms[case_id] )


        # tmp_terms : 医学考试中的重点概念
        # 在没有语义标签时，存放到 tmp_terms 中
        # 在具有语义标签后，存放到 pub_terms 中
        core_words_from_TmpTerms = get_corewords_from_TmpTerms(self.db_info_dict, self.ssu_type_of_tags)
        # 将这些核心词更新到 coreword_dict_cn
        coreword_dict_cn.update( core_words_from_TmpTerms )
        # print case
        case_id = list( core_words_from_TmpTerms.keys() ) [0]
        # print( "core_words_from_TmpTerms case ", case_id, core_words_from_TmpTerms[case_id] )


        # LATTE : observable_ent_name , observable_ent_ranges
        # 来自 LATTE 的 可观测或测量的 测量核心词 
        corewords_from_LatteKB = get_corewords_from_LatteKB(self.db_info_dict, self.ssu_type_of_tags)
        # 将这些核心词更新到 coreword_dict_cn
        coreword_dict_cn.update( corewords_from_LatteKB )
        # print case
        case_id = list( corewords_from_LatteKB.keys() ) [0]
        # print( "corewords_from_LatteKB case ", case_id, corewords_from_LatteKB[case_id] )



        #
        # 基于 coreword_dict_cn 构建 核心词扫描器
        # coreword_dict_cn = core_words_from_PubTerms + corewords_from_LatteKB
        # 关闭 case_sensitive=True
        self.coreword_processor_cn = KeywordProcessor4cn()
        self.coreword_processor_cn.add_keywords_from_dict( coreword_dict_cn )


        # [术语类属性词扫描器]
        # another flashtext keyword processor
        # build flashtext keyword processor based on def_ssu_std_values table
        # self.attrword_processor_cn = build_attribute_processor(self.db_info_dict, 'cn')
        # {"表型核心词||严重程度||轻度": [ "轻度" ]}
        dict_of_std_values = build_dict_of_std_values( self.info_of_std_attributes, 'cn' )
        # print( json.dumps(dict_of_std_values, ensure_ascii=False, indent=4) )

        # 构建术语类属性词扫描器
        # 关闭 case_sensitive=True 
        self.attrword_processor_cn = KeywordProcessor4cn()
        self.attrword_processor_cn.add_keywords_from_dict( dict_of_std_values )


        # 构建样本来源名称扫描器
        dict_of_sample_names = build_dict_of_std_values( self.info_of_sample_names, 'cn' )
        # 关闭 case_sensitive=True
        self.sample_name_processor_cn = KeywordProcessor4cn()
        self.sample_name_processor_cn.add_keywords_from_dict( dict_of_sample_names )


        # [ADD] 用于 可观测实体 的 正常性/异常性 判读
        # 构建 可观测实体 正常范围 知识库
        self.dict_of_ref_ranges = get_info_of_ref_ranges( self.db_info_dict )
        # print( json.dumps(self.dict_of_ref_ranges, ensure_ascii=False, indent=4) )


        # [ADD] 基于LATTE知识库，将 可观测实体名称 映射到 ent_first_name
        self.mapping_of_observable_ent_names = get_mapping_of_observable_ent_names( corewords_from_LatteKB )


        # [模式类属性词扫描器]
        # [NOTE][ADJUST] 
        # 不进行全文扫描而是在需要时围绕核心词展开 Context 扫描
        # 减少计算量的同时，代码的可理解性、可维护性、可扩展性更好


        # [Section 标题核心词(术语型)知识库]
        dict_of_section_triggers = get_section_triggers_info( self.db_info_dict )
        # print( "\n dict_of_section_triggers ", dict_of_section_triggers )

        # Section 标题词 扫描器
        # 关闭 case_sensitive=True
        self.section_name_processor_cn = KeywordProcessor4cn()
        self.section_name_processor_cn.add_keywords_from_dict( dict_of_section_triggers )


        # 术语的原子化(SSU)拆分知识库
        # 用于增强术语比对功能
        self.dict_of_term_splits_info = get_term_splits_info( self.db_info_dict )


        # N联征知识库
        # 用于增强术语比对功能
        self.dict_of_triad_term_info = get_term_triad_info( self.db_info_dict )



    #############
    # 类的功能函数
    #############

    # 函数名称: 测试函数
    # 函数功能: 测试输出
    def say_hello(self):
        # 
        # print('hello')
        print( self.coreword_processor_cn.extract_keywords("头晕头痛", span_info= True)  )

        #
        return 0


    ##############################################################
    # 文本标注函数
    # 输入: text_str
    # 输出: 文本中待标注的实体信息; 实体关联的属性或实体关联的实体信息
    # user_id参数的引入用于支持个性化标注
    ##############################################################
    def get_annotations_of_text(self, text_str, user_id):
        # [INPUT]
        # 给定一段文本

        # print( "0. 【文本标注函数】 get_annotations_of_text() 【最初输入】: text_str " )
        # print( text_str )


        # [Step 0]
        # [获取文本中的 Section Headers]
        # 目前主要考虑教科书的一级标题名称
        section_headers_found = get_section_headers_of_text( text_str, self.section_name_processor_cn )
        # print( "\n 1. 【文本Section扫描函数】 get_section_headers_of_text() 【中间输出】: section_headers_found " )
        # print( section_headers_found )


        # [STEP 1]
        # [扫描文本中的核心词]
        # [a. Pre-Processing]
        # 识别到标题后，屏蔽标题词，再识别核心词，避免潜在的 overlap
        # [BUG] 修复了 get_masked_text_str 无法正确生成 masked_text_str 的bug
        masked_text_str = get_masked_text_str( text_str, section_headers_found )
        # print("\nmasked_text_str\n ", masked_text_str)

        # [b. Recognizing core words from text]
        # [('C0012833||T184||Sign or Symptom', 0, 1), ('C0010200||T184||Sign or Symptom', 3, 4)]
        core_words_found = self.coreword_processor_cn.extract_keywords( masked_text_str, span_info=True )
        # print( "\n core_words_found ", core_words_found )
        
        # [c. Post-Preprocessing]
        # 对扫描到的核心词进行清洗，目前加入的主要清洗规则是 
        # 如果一个 keyword 不包含汉字, 则它要成为一个核心词的条件
        # 前一个字符和后一个字符不能是汉字、字母或数字等有意义的字符
        # 如 mmHg, Hg会被扫描为 血红蛋白 核心词, 这样的词需要被过滤
        core_words_found = filter_keywords_found( text_str, core_words_found )
        # print("\n 2. 【文本核心词扫描函数】 filter_keywords_found() 【中间输出】: core_words_found " )
        # print( core_words_found )

        # 增加一个将部位核心词标准化的功能
        core_words_found_std_body = []
        for core_word in core_words_found:
            body_span_info, body_span_start, body_span_end = core_word
            if core_word[0].split('||')[3] == '部位核心词':
                body_core_word = body_span_info.split('||')[0]
                body_core_word_info_std = normalize_term_to_cui_or_prefer( body_core_word, self.dict_of_terms_with_confirmed_cui, self.prefered_name_from_PubSyns )
                body_span_info_std = body_span_info.replace(body_core_word, body_core_word_info_std)
                core_word = (body_span_info_std, body_span_start, body_span_end)
            else:
                core_word = (body_span_info, body_span_start, body_span_end)
            core_words_found_std_body.append(core_word)
        core_words_found = core_words_found_std_body


        # [MEMO] 
        # 若需要加载用户的个性化词典, 只需要 mask 原来的标注, 然后使用用户自己的核心词处理器进行扫描
        # 若需要加入BERT-NER识别结果, 也可  mask 原来的标注, 然后使用BERT-NER的核心词扫描器进行识别


        # [Step 2]
        # [a. Pre-processing]
        # 属性词识别之前，需要结合 core_words_found 对 text_str 进行 mask 处理
        # 避免 overlap , 结节 (Finding) 结节状 (attributes)
        masked_text_str = get_masked_text_str( masked_text_str, core_words_found )

        # [b. 扫描文本中的属性词][基于mask后的文本进行扫描]
        # [('轻度||严重程度||表型核心词', 3, 4)]
        attr_words_found = self.attrword_processor_cn.extract_keywords( masked_text_str, span_info=True )

        # [c. 后处理]
        attr_words_found = filter_keywords_found( text_str, attr_words_found )
        # print("\n 3. 【文本属性词扫描函数】 filter_keywords_found() 【中间输出】: attr_words_found " )
        # print( attr_words_found )
# # 

        # 进一步屏蔽文本中识别得到的属性词
        # 这样, masked_text_str, 屏蔽了标题词、核心词和属性词
        masked_text_str = get_masked_text_str( masked_text_str, attr_words_found )
        # print(masked_text_str)

        # [2.2 扫描文本中的属性模式]
        # [MEMO] Suspend


        # [Step 3]
        # 3.1 获取SSU核心词关联的属性词
        # 关联方法: 确定非一般核心词如表型核心词的SCOPE, 把 SCOPE[0] 和 SCOPE[1] 放入到 attr_words_found 列表
        # 确定 SCOPE[0] 和 SCOPE[1] 在 attr_words_found 中得位置, 截取SCOPE区间内的属性词，与核心词进行关联
        ent_related_attr_words = get_ent_related_attr_words(text_str, core_words_found, attr_words_found, self.info_of_ssu_attributes)
        # print("\n 4. 【核心词-属性词关联函数】 get_ent_related_attr_words() 【中间输出】: ent_related_attr_words " )
        # print( ent_related_attr_words )


        # 3.2 获取SSU核心词关联的核心词
        ent_related_core_words = get_ent_related_core_words(text_str, core_words_found, self.dict_of_terms_with_confirmed_cui, self.prefered_name_from_PubSyns)
        # print("\n 5. 【核心词-核心词关联函数】 get_ent_related_core_words() 【中间输出】: ent_related_core_words " )
        # print( ent_related_core_words )



        # [OUTPUT]
        annotated_info_dict = {}
        # ent_info to be annotated
        annotated_ent_info = {}
        # rel_info to be annotated
        annotated_rel_info = {}

        # dict_of_ann_info.setdefault("实体标注信息", dict_of_ent_info)
        # dict_of_ann_info.setdefault("实体关系信息", dict_of_rel_info)


        # [获取文本中需要标注的实体信息 ent_info]
        # 核心词实体均需要标注
        # 与核心词关联的属性词(术语型或模式型)实体需要标注 (但平时处于隐藏状态)
        # 两段数据，后端数据写在函数中; 返回给前端的数据可写在接口中，函数之后，返回之前。
        annotated_ent_info = get_annotated_ent_info( text_str, section_headers_found, 
                                                    core_words_found, attr_words_found, ent_related_attr_words )
        # print( "\n 6. 【文本中待标注实体信息整理函数】 get_annotated_ent_info() 【中间输出】: annotated_ent_info " )
        # print( annotated_ent_info ) 


        # [获取文本中需要标注的关系信息 rel_info]
        # 实体包含的属性信息
        # 实体关联的实体信息
        # 数据结构 中心实体-关联实体  
        # core_ent_id: [ (related_ent_id, "关系名称") ]
        annotated_rel_info = get_annotated_rel_info( ent_related_attr_words, ent_related_core_words )
        # print( "\n 7. 【文本中待标注关系信息整理函数】 get_annotated_rel_info() 【中间输出】: annotated_rel_info " )
        # print( annotated_rel_info )



        # 存储实体标注信息和实体关系信息
        annotated_info_dict.setdefault("annotated_ent_info", annotated_ent_info)
        annotated_info_dict.setdefault("annotated_rel_info", annotated_rel_info)


        # [UPDATE]
        # 升级文本中 表型类实体 的 持续时间 阅读功能
        # 函数输出: updated_annotated_info_dict
        # [Add] 升级文本中 表型类实体 的 既往存在 属性理解能力        
        annotated_info_dict = update_info_of_phenotype_ents( masked_text_str, annotated_info_dict )
        # print( "\n 8. 【文本中表型类实体信息更新函数】 update_info_of_phenotype_ents() 【中间输出】: updated_annotated_info_dict " )
        # print( annotated_info_dict )


        # [UPDATE]
        # 升级文本中 可观测实体 的 正常性/异常性 判断功能
        # 搜索 annotated_ent_info 中的 可观测实体
        # 确立 该可观测实体 的 left_context, 使用 样本名称扫描器 扫描 样本来源 
        # 如果 样本来源 是血液，正常填入实体信息；但在查询 LATTE 知识库时转为NULL
        # 如果 没有扫描到任何样本来源，样本来源字段设置为 'NULL'
        # 获取 该可观测实体 的 restricted_units，在 right_context 中扫描限定单位
        # 如果能扫描到 限定单位, 提取 可观测实体 到 限定单位 之间的数据进行判断
        # 如果没扫描到 限定单位，且 该可观测实体 存在无需单位的观测值。观测 right_context 中除了数值之外没有别的东西。
        # 如果没有，则需要将 观测值 提取为一个 ent, 并加之加入到 annotated_ent_info 和 annotated_rel_info 字典中。
        # 函数名称: 基于 obserable_ents 生成 语义结构单元
        # 函数输出: updated_annotated_info_dict
        annotated_info_dict = update_info_of_observable_ents( masked_text_str, annotated_info_dict, self.sample_name_processor_cn, 
                                                            self.mapping_of_observable_ent_names, self.dict_of_ref_ranges)

        # print( "\n 9. 【文本中测量类实体信息更新函数】 update_info_of_observable_ents() 【中间输出】: updated_annotated_info_dict " )
        # print( annotated_info_dict )



        # [UPDATE]
        # 升级文本中 药物类实体 的 关联属性 阅读功能
        # 函数输出: updated_annotated_info_dict
        # annotated_info_dict = update_info_of_drug_ents( text_str, annotated_info_dict )
        # print( "\n update_info_of_drug_ents() output: updated_annotated_info_dict ", annotated_info_dict )        
        annotated_info_dict = update_info_of_drug_ents( masked_text_str, annotated_info_dict )
        # print( "\n 10. 【文本中药物类实体信息更新函数】 update_info_of_drug_ents() 【中间输出】: updated_annotated_info_dict " )
        # print( annotated_info_dict )

        #
        # Final Output
        # print( "\n 11. [文本标注函数最终输出] get_annotations_of_text() 【最终输出】: annotated_info_dict" )
        # print( json.dumps( annotated_info_dict, ensure_ascii=False, indent=4 ) )


        #
        return annotated_info_dict


    ##########################
    # 文本标注结果的格式转换函数
    ##########################
    # 将文本标注结果转换为 易于阅读的形式 格式
    # 输出 以核心词为中心 的标注结果
    def convert_annotations_to_human_readable_data(self, annotated_info_dict):
        # 输出以核心词为中心的标注结果
        human_readable_info = {}

        # 获取标注数据
        annotated_ent_info = annotated_info_dict["annotated_ent_info"]
        annotated_rel_info = annotated_info_dict["annotated_rel_info"]        

        # 
        for ent_id in annotated_ent_info:
            #
            ent_info = annotated_ent_info[ent_id]

            # 
            if ent_info["term_type"] == '核心词':
                # 观察该核心词是否有关联的属性信息
                # 关联的实体信息
                related_ent_info = []

                if ent_id in annotated_rel_info:
                    for rel_name_str, rel_ent_id in annotated_rel_info[ent_id]:
                        related_ent_info.append( annotated_ent_info[rel_ent_id] )

                # 在 ent_info 中增加 related_ents 字段
                ent_info.setdefault( "related_ents", related_ent_info)

                # store
                human_readable_info.setdefault( ent_id, ent_info )


        return human_readable_info


    # 将文本标注转为 jstree 的数据结构
    def convert_annotations_to_jstree_data(self, annotated_info_dict):
        # 方法
        # 遍历 实体列表 , 将其先全部挂在 Root 结点下
        # 更改 核心词实体的 Parent Node, 更改为邻近的标题型结点
        # 更改 属性词实体的 Parent Node, 更改为关联的核心词结点

        # 求解目标
        # [ { "id": "root_node", "text": "文档知识图谱", "parent": "#" }, 
        # { "id": "0:3", "text": "【概述】", "parent": "root_node" }, 
        # { "id": "7:8", "text": "头痛", "parent": "0:3" } ]
        # 需要考虑结点复用的问题，所以不能用 span_key 作为 node_id
        # jstree 结点列表
        list_of_jstree_nodes = []
        # 用于记录结点在文本中的ent_id [便于定位某node并修改某node的信息]
        ent_ids_of_jstree_nodes = []


        # 获取标注数据
        annotated_ent_info = annotated_info_dict["annotated_ent_info"]
        annotated_rel_info = annotated_info_dict["annotated_rel_info"]

        # [PATCH]
        # 如果一个核心词在 annotated_rel_info 是依附于 某核心词 (表型: 发作部位, 部位核心词)
        # 暂时先不要将它加入到 jstree 结点
        ent_ids_as_attributes = get_ent_ids_as_attributes( annotated_rel_info )
        # print(" ent_ids_as_attributes ", ent_ids_as_attributes)

        # 为便于合并核心词的同类属性
        # 更改 annotated_rel_info 的存储形式
        # core_ent_id: {"关系名称":[], "关系名称":[]}
        grouped_annotated_rel_info = group_annotated_rel_info_by_rel_name( annotated_ent_info, annotated_rel_info )
        # print("\n grouped_annotated_rel_info ", grouped_annotated_rel_info)


        # 设置 jstree 根节点
        # note_info
        root_node_info = {}
        # id required
        # 根据在 list_of_jstree_nodes 中的 结点数目 自增
        # 意味着 在 jstree 中新增一个结点 (node_id = 0 代表根结点)
        jstree_node_id = 0
        root_node_info.setdefault( "id", str(jstree_node_id) )
        # node text
        root_node_info.setdefault( "text", "思维导图 By AI" )        
        # parent required        
        # To indicate a node should be a root node set its parent property to "#".
        root_node_info.setdefault( "parent", "#" )
        # span in the text 
        root_node_info.setdefault( "ent_id", "" )
        # term type 
        root_node_info.setdefault( "ent_type", "核心词" )
        # store root node
        list_of_jstree_nodes.append( root_node_info )
        # store the coresponding ent ids of jstree node 
        ent_ids_of_jstree_nodes.append( "" )
        # node_ides_of_jstree_nodes.append( jstree_node_id )


        # 遍历 实体列表 , 将标题词和核心词 其先全部挂在 Root 结点下        
        # 需要 ent_id, ent_str, "#"
        # 同时 基于 annotated_ent_info 生成一条 seq_of_annotated_ent_info 的单字母序列
        seq_of_annotated_ents = ""

        for ent_id in annotated_ent_info:
            #
            ent_info = annotated_ent_info[ent_id]

            # term_str
            # print( "ent_info " , ent_info)
            ent_str = ent_info['term_str']


            # term_type 标题词 图片词 核心词 属性词
            ent_type = ent_info["term_type"]

            # 将标题词和核心词 其先全部挂在 Root 结点下
            # 部位核心词可作为属性依附于表型核心词
            # 不考虑依附了某表型核心词的部位核心词            
            if ent_type in ["标题词", '核心词'] and ent_id not in ent_ids_as_attributes:
                ent_node_info = {}                
                # node id  +1 表示在 jstree 中新增一个结点
                jstree_node_id += 1
                ent_node_info.setdefault( "id", str(jstree_node_id) )
                #
                ent_node_info.setdefault( "text", ent_str )
                # parent (root node id = 0)
                ent_node_info.setdefault( "parent", 0 )   
                # span
                ent_node_info.setdefault( "ent_id", ent_id )
                # type
                # 标题词也视为一种核心词
                ent_node_info.setdefault( "ent_type", "核心词" )
                # store ent node
                list_of_jstree_nodes.append( ent_node_info ) 
                # store node ent id
                ent_ids_of_jstree_nodes.append( ent_id )   
                # node_ides_of_jstree_nodes.append( jstree_node_id )                


            # 基于 ent 生成标题、核心词、属性词的实体类型单字母表征的序列
            # 以便于寻找某核心词所依附的标题词
            if ent_type == '标题词':
                seq_of_annotated_ents += 'T'
            elif ent_type == '核心词':
                seq_of_annotated_ents += 'C'
            elif ent_type == '属性词':
                seq_of_annotated_ents += 'A'     


        # 中间结果
        # [{'id': '1', 'text': '思维导图 By AI', 'parent': '#', 'ent_id': ''}, 
        # {'id': '2', 'text': '【症状】', 'parent': '#', 'ent_id': '0:3'}, 
        # {'id': '3', 'text': '头晕', 'parent': '#', 'ent_id': '5:6'}]
        # print( "convert_annotations_to_jsmind_data 中间结果: ",  list_of_jstree_nodes)
        # TCA
        # print( seq_of_annotated_ents )
        # ['', '0:3', '5:6']
        # print( ent_ids_of_jstree_nodes )


        # 遍历 实体列表中的 核心词, 寻找离它最近的 标题词
        # 更改 dict_of_jstree_info[ent_id]中 ent_node_info['parent']为该标题词的 ent_id
        # 
        # 遍历 实体列表中的 核心词, 如果它有关联的属性词或实体词，将之作为子结点置于该核心词下
        list_of_annotated_ents = list( annotated_ent_info.keys() )

        for ent_idx, ent_id in enumerate(list_of_annotated_ents):
            #
            ent_info = annotated_ent_info[ent_id]
            ent_type = ent_info["term_type"]

            #
            if ent_type == '核心词' and ent_id not in ent_ids_as_attributes:
                # a. 遍历 实体列表中的 核心词, 寻找离它最近的 标题词
                # 基于 seq_of_annotated_ents 寻找它所依附的标题词 (最近邻规则)
                # 从 seq_of_annotated_ents 的 第 ent_idx 位置 逆序查找第一个标题词
                title_ent_idx = seq_of_annotated_ents.rfind( 'T', 0, ent_idx )
                # 0
                # print("title_ent_idx in seq_of_annotated_ents:", title_ent_idx )

                # 如果存在这样一个标题词, 更新 list_of_jstree_nodes 结点数据
                if title_ent_idx != -1:
                    # 基于 标题的idx 提取 该标题词的 ent_id = list_of_annotated_ents[title_ent_idx]
                    title_ent_id = list_of_annotated_ents[title_ent_idx]
                    # 0:3
                    # print( "title_ent_id ",  title_ent_id )

                    # 更改 list_of_jstree_info 中 该核心词 的 parent node
                    # 先定位 该核心词 在 list_of_jstree_info 中的位置
                    if ent_id in ent_ids_of_jstree_nodes:
                        # 获取 该核心词 在 ent_ids_of_jstree_nodes 列表中的结点位置 和 结点信息
                        ent_node_idx = ent_ids_of_jstree_nodes.index( ent_id )
                        # 基于该位置 从 list_of_jstree_nodes 中提取 ent_node_info
                        ent_node_info = list_of_jstree_nodes[ent_node_idx]

                        # 更改 结点 ent_node_info的 parent 信息为关联的标题
                        if title_ent_id in ent_ids_of_jstree_nodes:
                            # 获取 标题核心词 在 jstree 中的 node_id
                            title_node_id = ent_ids_of_jstree_nodes.index( title_ent_id )
                            # 更新 该核心词 的parent id 为关联标题核心词对应的 node_id
                            ent_node_info['parent'] = title_node_id
                            # 保存到 list_of_jstree_nodes 结点信息列表
                            list_of_jstree_nodes[ent_node_idx] = ent_node_info


                # b. 遍历 实体列表中的 核心词, 将其关联的属性词或实体词挂到该核心词下面
                # core_ent_id: [ (related_ent_id, "关系名称") ]

                # 将核心词其关联的属性词或实体词挂到该核心词下面
                if ent_id in grouped_annotated_rel_info:
                    # 获取与该核心词关联的属性词或实体词 ent_id
                    for attr_name_str in grouped_annotated_rel_info[ent_id]:
                        # rel_name_str 对应的 attr_word_ids
                        attr_word_ids = grouped_annotated_rel_info[ent_id][attr_name_str]

                        # 生成 jstree data                    
                        attr_node_info = {}

                        # 在jstree中新增一个结点
                        jstree_node_id += 1
                        attr_node_info.setdefault( 'id', str(jstree_node_id) )


                        # text
                        # 从 annotated_ent_info 提取数据
                        # annotated_ent_info.setdefault(ent_id, ent_info)
                        # 属性名称 + 属性列表
                        # 发作部位: 手部、足部、口部
                        attr_word_texts = []

                        for attr_word_id in attr_word_ids:
                            # 
                            attr_word_info = annotated_ent_info[attr_word_id]
                            # term_str
                            attr_word_texts.append( attr_word_info['term_str'] )


                        # rel_name_str has_attribute
                        attr_node_text = attr_name_str + ': ' + ', '.join( attr_word_texts )
                        attr_node_info.setdefault( "text", attr_node_text)

                        # 设置 所关联的核心词 为 parent 
                        # default, 挂到 root 结点
                        attr_node_info.setdefault( "parent", 0)

                        if ent_id in ent_ids_of_jstree_nodes:
                            # 所关联的核心词在 jstree 上的node_id
                            ent_node_id = ent_ids_of_jstree_nodes.index( ent_id )
                            # 更新 parent id 
                            attr_node_info['parent'] = ent_node_id

                        # ent_id of the node : 如果有多个属性值，取第一个属性实体的 attr_word_id
                        # 这意味着当在jstree中点击该结点，可跳转到文本中对应位置
                        attr_node_id = attr_word_ids[0]
                        attr_node_info.setdefault( 'ent_id', attr_node_id )
                        # type
                        attr_node_info.setdefault( 'ent_type', "属性词" )

                        # store attr node
                        list_of_jstree_nodes.append( attr_node_info )
                        # store the ent id correponding to this node
                        ent_ids_of_jstree_nodes.append( attr_node_id )

        #
        return list_of_jstree_nodes


    # 将文本标注转为 jsMind 的数据结构
    # 和 jsTree 类似，存储字段名称变变
    def convert_annotations_to_jsmind_data(self, annotated_info_dict):
        # 方法
        # 遍历 实体列表 , 将其先全部挂在 Root 结点下
        # 更改 核心词实体的 Parent Node, 更改为邻近的标题型结点
        # 更改 属性词实体的 Parent Node, 更改为关联的核心词结点

        # 求解目标
        # [ { "id": "root_node", "text": "文档知识图谱", "parent": "#" }, 
        # { "id": "0:3", "text": "【概述】", "parent": "root_node" }, 
        # { "id": "7:8", "text": "头痛", "parent": "0:3" } ]
        # 需要考虑结点复用的问题，所以不能用 span_key 作为 node_id
        # jstree 结点列表
        # [MEMO] 名字不用改了，这样方便点
        list_of_jstree_nodes = []
        # 用于记录结点在文本中的ent_id [便于定位某node并修改某node的信息]
        ent_ids_of_jstree_nodes = []


        # 获取标注数据
        annotated_ent_info = annotated_info_dict["annotated_ent_info"]
        annotated_rel_info = annotated_info_dict["annotated_rel_info"]

        # [PATCH]
        # 如果一个核心词在 annotated_rel_info 是依附于 某核心词 (表型: 发作部位, 部位核心词)
        # 暂时先不要将它加入到 jstree 结点
        ent_ids_as_attributes = get_ent_ids_as_attributes( annotated_rel_info )

        # 为便于合并核心词的同类属性
        # 更改 annotated_rel_info 的存储形式
        # core_ent_id: {"关系名称":[], "关系名称":[]}
        grouped_annotated_rel_info = group_annotated_rel_info_by_rel_name( annotated_rel_info )


        # 设置 jstree 根节点
        # note_info
        root_node_info = {}
        # id required
        # 根据在 list_of_jstree_nodes 中的 结点数目 自增
        # 意味着 在 jstree 中新增一个结点 (node_id = 0 代表根结点)
        jstree_node_id = 0
        root_node_info.setdefault( "id", str(jstree_node_id) )
        # node text
        root_node_info.setdefault( "topic", "思维导图 By AI" )        
        # parent required        
        # To indicate a node should be a root node set its parent property to "#".
        root_node_info.setdefault( "isroot", "true" )
        # span in the text 
        root_node_info.setdefault( "ent_id", "" )
        # store root node
        list_of_jstree_nodes.append( root_node_info )
        # store the coresponding ent ids of jstree node 
        ent_ids_of_jstree_nodes.append( "" )
        # node_ides_of_jstree_nodes.append( jstree_node_id )


        # 遍历 实体列表 , 将标题词和核心词 其先全部挂在 Root 结点下
        # 需要 ent_id, ent_str, "#"
        # 同时 基于 annotated_ent_info 生成一条 seq_of_annotated_ent_info 的单字母序列
        seq_of_annotated_ents = ""

        for ent_id in annotated_ent_info:
            #
            ent_info = annotated_ent_info[ent_id]

            # term_str
            # print( "ent_info " , ent_info)
            ent_str = ent_info['term_str']


            # term_type 标题词 核心词 属性词
            ent_type = ent_info["term_type"]

            # 将标题词和核心词 其先全部挂在 Root 结点下
            # 部位核心词可作为属性依附于表型核心词
            # 不考虑依附了某表型核心词的部位核心词            
            if ent_type in ["标题词", '核心词'] and ent_id not in ent_ids_as_attributes:
                ent_node_info = {}                
                # node id  +1 表示在 jstree 中新增一个结点
                jstree_node_id += 1
                ent_node_info.setdefault( "id", str(jstree_node_id) )
                #
                ent_node_info.setdefault( "topic", ent_str )
                # parent (root node id = 0)
                ent_node_info.setdefault( "parentid", 0 )   
                # span
                ent_node_info.setdefault( "ent_id", ent_id )
                # store ent node
                list_of_jstree_nodes.append( ent_node_info ) 
                # store node ent id
                ent_ids_of_jstree_nodes.append( ent_id )   
                # node_ides_of_jstree_nodes.append( jstree_node_id )                


            # 基于 ent 生成标题、核心词、属性词的实体类型单字母表征的序列
            # 以便于寻找某核心词所依附的标题词
            if ent_type == '标题词':
                seq_of_annotated_ents += 'T'
            elif ent_type == '核心词':
                seq_of_annotated_ents += 'C'
            elif ent_type == '属性词':
                seq_of_annotated_ents += 'A'     


        # 中间结果
        # [{'id': '1', 'text': '思维导图 By AI', 'parent': '#', 'ent_id': ''}, 
        # {'id': '2', 'text': '【症状】', 'parent': '#', 'ent_id': '0:3'}, 
        # {'id': '3', 'text': '头晕', 'parent': '#', 'ent_id': '5:6'}]
        # print( "convert_annotations_to_jsmind_data 中间结果: ",  list_of_jstree_nodes)
        # TCA
        # print( seq_of_annotated_ents )
        # ['', '0:3', '5:6']
        # print( ent_ids_of_jstree_nodes )


        # 遍历 实体列表中的 核心词, 寻找离它最近的 标题词
        # 更改 dict_of_jstree_info[ent_id]中 ent_node_info['parent']为该标题词的 ent_id
        # 
        # 遍历 实体列表中的 核心词, 如果它有关联的属性词或实体词，将之作为子结点置于该核心词下
        list_of_annotated_ents = list( annotated_ent_info.keys() )

        for ent_idx, ent_id in enumerate(list_of_annotated_ents):
            #
            ent_info = annotated_ent_info[ent_id]
            ent_type = ent_info["term_type"]

            #
            if ent_type == '核心词' and ent_id not in ent_ids_as_attributes:
                # a. 遍历 实体列表中的 核心词, 寻找离它最近的 标题词
                # 基于 seq_of_annotated_ents 寻找它所依附的标题词 (最近邻规则)
                # 从 seq_of_annotated_ents 的 第 ent_idx 位置 逆序查找第一个标题词
                title_ent_idx = seq_of_annotated_ents.rfind( 'T', 0, ent_idx )
                # 0
                # print("title_ent_idx in seq_of_annotated_ents:", title_ent_idx )

                # 如果存在这样一个标题词, 更新 list_of_jstree_nodes 结点数据
                if title_ent_idx != -1:
                    # 基于 标题的idx 提取 该标题词的 ent_id = list_of_annotated_ents[title_ent_idx]
                    title_ent_id = list_of_annotated_ents[title_ent_idx]
                    # 0:3
                    # print( "title_ent_id ",  title_ent_id )

                    # 更改 list_of_jstree_info 中 该核心词 的 parent node
                    # 先定位 该核心词 在 list_of_jstree_info 中的位置
                    if ent_id in ent_ids_of_jstree_nodes:
                        # 获取 该核心词 在 ent_ids_of_jstree_nodes 列表中的结点位置 和 结点信息
                        ent_node_idx = ent_ids_of_jstree_nodes.index( ent_id )
                        # 基于该位置 从 list_of_jstree_nodes 中提取 ent_node_info
                        ent_node_info = list_of_jstree_nodes[ent_node_idx]

                        # 更改 结点 ent_node_info的 parent 信息为关联的标题
                        if title_ent_id in ent_ids_of_jstree_nodes:
                            # 获取 标题核心词 在 jstree 中的 node_id
                            title_node_id = ent_ids_of_jstree_nodes.index( title_ent_id )
                            # 更新 该核心词 的parent id 为关联标题核心词对应的 node_id
                            ent_node_info['parentid'] = title_node_id
                            # 保存到 list_of_jstree_nodes 结点信息列表
                            list_of_jstree_nodes[ent_node_idx] = ent_node_info


                # b. 遍历 实体列表中的 核心词, 将其关联的属性词或实体词挂到该核心词下面
                # core_ent_id: [ (related_ent_id, "关系名称") ]

                # 将核心词其关联的属性词或实体词挂到该核心词下面
                if ent_id in grouped_annotated_rel_info:
                    # 获取与该核心词关联的属性词或实体词 ent_id
                    for rel_name_str in grouped_annotated_rel_info[ent_id]:
                        # rel_name_str 对应的 attr_word_ids
                        attr_word_ids = grouped_annotated_rel_info[ent_id][rel_name_str]

                        # 生成 jstree data                    
                        attr_node_info = {}

                        # 在jstree中新增一个结点
                        jstree_node_id += 1
                        attr_node_info.setdefault( 'id', str(jstree_node_id) )


                        # text
                        # 从 annotated_ent_info 提取数据
                        # annotated_ent_info.setdefault(ent_id, ent_info)
                        # 属性名称 + 属性列表
                        # 发作部位: 手部、足部、口部
                        attr_word_texts = []
                        attr_name_str = "关联属性"

                        for attr_word_id in attr_word_ids:
                            # 
                            attr_word_info = annotated_ent_info[attr_word_id]
                            # term_str
                            attr_word_texts.append( attr_word_info['term_str'] )

                            # 如果是属性词实体
                            if "attr_name" in attr_word_info:
                                attr_name_str = attr_word_info["attr_name"]

                            # 如果是部位核心词实体
                            if "core_type" in attr_word_info:
                                if attr_word_info['core_type'] == '部位核心词':
                                    attr_name_str = '发作部位'


                        # rel_name_str has_attribute
                        attr_node_text = attr_name_str + ': ' + ', '.join( attr_word_texts )
                        attr_node_info.setdefault( "topic", attr_node_text)

                        # 设置 所关联的核心词 为 parent 
                        # default, 挂到 root 结点
                        attr_node_info.setdefault( "parentid", 0)

                        if ent_id in ent_ids_of_jstree_nodes:
                            # 所关联的核心词在 jstree 上的node_id
                            ent_node_id = ent_ids_of_jstree_nodes.index( ent_id )
                            # 更新 parent id 
                            attr_node_info['parentid'] = ent_node_id

                        # ent_id of the node : 如果有多个属性值，取第一个属性实体的 attr_word_id
                        # 这意味着当在jstree中点击该结点，可跳转到文本中对应位置
                        attr_node_id = attr_word_ids[0]
                        attr_node_info.setdefault( 'ent_id', attr_node_id )

                        # store attr node
                        list_of_jstree_nodes.append( attr_node_info )
                        # store the ent id correponding to this node
                        ent_ids_of_jstree_nodes.append( attr_node_id )

        #
        return list_of_jstree_nodes


    #################################################################################
    # 文本比对函数 (高级文本比对)
    # 输入: 文本a, 文本b
    # 输出: 两段文本的比对结果        
    # 参考序列: 文本a 
    # 若给定文本a中的某个ent, 能给出文本b中能够对应上的ent list)
    # 当点击文本a中的某个ent, 显示文本b中所有能够对应上的 ent , 定位到能够匹配上的第一个ent
    # 对应关系分为两种: 完全相同，部分相似
    # [MEMO] 相似程度扩展 相似程度百分比  2*set(a)&set(b)/[set(a) + set(b)]
    # [NOTE] 引入 user_id 参数是考虑到后续扩展个性化标注的需求
    # [PATCH] 返回 text_a, text_b 的 文本标注信息
    # [PATCH] 如果 text_a中 某核心词的其子结点 可以与 text_b中某核心词比对上 
    # 发热 -- 高热 (部分相似)  肺炎 -- 非小泡肺炎 (部分相似)
    ##################################################################################
    def get_ssu_alignment_btw_texts(self, text_str_a, text_str_b, user_id ):
        # 求解目标
        # ent_id_in_a, [("完全相同", ent_id_in_b), ("部分相似", ent_id_in_b) ]
        dict_of_alignment_results = {}

        #
        # print( "input text_str_a: ", text_str_a)
        # print( "input text_str_b: ", text_str_b)

        # 比对方法
        # a.
        # text_str_a, text_str_b -> get_annotations_of_text 
        # ent_info and rel_info in text a and text b
        annotated_info_of_text_a = self.get_annotations_of_text( text_str_a, user_id )
        ent_info_in_text_a = annotated_info_of_text_a["annotated_ent_info"]
        rel_info_in_text_a = annotated_info_of_text_a["annotated_rel_info"]

        annotated_info_of_text_b = self.get_annotations_of_text( text_str_b, user_id )
        ent_info_in_text_b = annotated_info_of_text_b["annotated_ent_info"]
        rel_info_in_text_b = annotated_info_of_text_b["annotated_rel_info"]


        
        # b. 
        # 遍历 ent_info_in_text_a , for a certain ent
        # 生成 binary_std_exp_set 二元标准化表达集合
        # 核心词的CUI::属性词的标准值
        # '5:6': {'span_start': 5, 'span_end': 6, 'term_type': '核心词', 'term_str': '头晕',
        # 'term_cui': 'C0012833', 'term_tui': 'T184', 'term_sty': 'Sign or Symptom', 'core_type': '表型核心词'}

        # 获取 text_a, text_b 的二元标准化表达集合
        binary_expressions_of_text_a = convert_annotations_to_binary_expressions( annotated_info_of_text_a )
        binary_expressions_of_text_b = convert_annotations_to_binary_expressions( annotated_info_of_text_b )


        ent_a_raw_core_list = []
        ent_b_raw_core_list = []
        ent_a_attr_list_all = []
        ent_b_attr_list_all = []
        
        ent_id_a_list = []
        ent_id_b_list = []
        align_info_btw_attr_list = []
        
        ent_a_core_type_list  = []
        ent_b_core_type_list  = []
        for ent_id_a in binary_expressions_of_text_a:
            ent_a_core_type = ent_info_in_text_a[ent_id_a]['core_type']
            #
            ent_a_attr_list  = binary_expressions_of_text_a[ent_id_a]['binary_exp_list']
            
            #
            ent_a_raw_core = binary_expressions_of_text_a[ent_id_a]['raw_coreterm_str']
            
            # yangtao_patch
            # 输出结果补充进属性值
            # 找到属性词的标准取值和id
            # 最终结果为一个大的列表里面是属性id和属性标准值的小字典
            core_attr_id_stdvalue_a = core_attr_id_stdvalue(ent_info_in_text_a, rel_info_in_text_a, ent_id_a, binary_expressions_of_text_a)
            # 
            for ent_id_b in binary_expressions_of_text_b:
                ent_b_core_type = ent_info_in_text_b[ent_id_b]['core_type']

                ent_b_attr_list  = binary_expressions_of_text_b[ent_id_b]['binary_exp_list']

                #
                ent_b_raw_core = binary_expressions_of_text_b[ent_id_b]['raw_coreterm_str']

                core_attr_id_stdvalue_b = core_attr_id_stdvalue(ent_info_in_text_b, rel_info_in_text_b, ent_id_b, binary_expressions_of_text_b)

                # 找到相似的表型术语的相似属性位置
                align_info_btw_attr = align_core_attr_id_stdvalue(core_attr_id_stdvalue_a, core_attr_id_stdvalue_b)

                align_info_btw_attr_list.append(align_info_btw_attr)
                ent_a_raw_core_list.append(ent_a_raw_core)
                ent_b_raw_core_list.append(ent_b_raw_core)
                ent_a_attr_list_all.append(ent_a_attr_list)
                ent_b_attr_list_all.append(ent_b_attr_list)
                ent_id_a_list.append(ent_id_a)
                ent_id_b_list.append(ent_id_b)
                ent_a_core_type_list.append(ent_a_core_type)
                ent_b_core_type_list.append(ent_b_core_type)

        # 按照batch输入到核心词判断的模型
        similar_categories = Term_judging_similar_categories(ent_a_raw_core_list, ent_b_raw_core_list)

        semantic_similar_indices = [i for i, category in enumerate(similar_categories) if category != "不相似"]

        for index in semantic_similar_indices:
            core_term_similar_categories = similar_categories[index]
            core_type_a= ent_a_core_type_list[index]
            core_type_b = ent_b_core_type_list[index]

            attr_list_a = ent_a_attr_list_all[index]
            attr_list_b = ent_b_attr_list_all[index]

            attr_set_similar_categories = attribute_set_similar_judgement(attr_list_a, attr_list_b)
            
            align_info_btw_ents = ssu_similar_categories(core_type_a, core_type_b, core_term_similar_categories, attr_set_similar_categories)
            core_id_a = ent_id_a_list[index]
            core_id_b = ent_id_b_list[index]

            similar_btw_attr_id = align_info_btw_attr_list[index]
        #         # 判断ssu之间相似类型的列表
        #         align_info_btw_ents = []
        #         # 首先是判断核心词的相似类别
        #         if ent_a_raw_core == ent_b_raw_core:
        #             core_term_similar_categories = '完全相等'
        #             attr_set_similar_categories = attribute_set_similar_judgement(ent_a_attr_list, ent_b_attr_list)
        #             align_info_btw_ents = ssu_similar_categories(ent_a_core_type, ent_b_core_type, core_term_similar_categories, attr_set_similar_categories)
        #         else:
        #             core_term_similar_categories = Term_judging_similar_categories(ent_a_raw_core, ent_b_raw_core)
        #             if core_term_similar_categories == '不相似':
        #                 align_info_btw_ents = ['不相似']
        #             else:
        #                 # 然后是是判断属性集合的相似类别
        #                 attr_set_similar_categories = attribute_set_similar_judgement(ent_a_attr_list, ent_b_attr_list)
        #                 align_info_btw_ents = ssu_similar_categories(ent_a_core_type, ent_b_core_type, core_term_similar_categories, attr_set_similar_categories)     
            if core_type_a in ["测量核心词", '一般核心词', '表型核心词'] and core_type_b in ["测量核心词", '一般核心词', '表型核心词']:
                if align_info_btw_ents[0] != '不相似':
                    align_info_btw_ents.append( core_id_b )
                    # align_info_btw_ents.append( similar_btw_attr_id[0] )
                else:
                    align_info_btw_ents = []

                # 可比对的时候，才store 
                if len(align_info_btw_ents) != 0:
                    if core_id_a not in dict_of_alignment_results:
                        dict_of_alignment_results.setdefault( core_id_a, [] )
                    #
                    dict_of_alignment_results[core_id_a].append( align_info_btw_ents )


        dict_of_align_info = {}
        dict_of_align_info.setdefault( "alignment_results", dict_of_alignment_results )
        #
        dict_of_align_info.setdefault( "text_a_annotations", annotated_info_of_text_a )
        dict_of_align_info.setdefault( "text_b_annotations", annotated_info_of_text_b)
        #
        return dict_of_align_info


    ############################
    # 文本核心词比对函数 (极速版)
    # 输入: 文本a, 文本b
    # 输出: 两端文本中相同的核心词
    # 建立核心词之间的映射关系
    # 完全相等，部分相似
    # [NOTE] 不需要带有坐标
    # [PATCH] 同义词比对; 父子结点比对
    # [UPDATE] 引入术语的原子化拆分信息 (如果核心词相同，比对上)
    ############################
    def get_core_alignment_btw_texts(self, text_str_a, text_str_b, user_id ):
        # 求解目标
        # { keyword_in_a: [keyword_in_b, keyword_in_b] }
        dict_of_align_info = {}

        #
        print( "input text_str_a: ", text_str_a)
        print( "input text_str_b: ", text_str_b)

        # 在 text_str_a 中比对上的 核心词
        # 在 text_str_b 中比对上的 核心词 
        # relation 映射关系

        # 扫描 text_str_a 中的核心词
        # ('痰多||T184||Sign or Symptom||表型核心词'
        # 改造后，pub_syns的同义词已经融合到 pub_terms生成的 self.coreword_processor_cn 中了
        core_words_found_in_a = self.coreword_processor_cn.extract_keywords( text_str_a, span_info=True )
        # 调用 filter 函数
        core_words_found_in_a = filter_keywords_found( text_str_a, core_words_found_in_a )
        # print( "core_words_found_in_a ", text_str_a, core_words_found_in_a)

        # 获取核心词的标准值作为 key
        dict_of_core_words_in_a = {}

        #
        for core_word_info in core_words_found_in_a:
            #
            span_info, span_start, span_end = core_word_info

            # info
            term_cui, term_tui, term_sty, core_type = span_info.split("||")

            # 
            term_str = text_str_a[span_start:span_end+1]

            # 
            if term_cui not in dict_of_core_words_in_a:
                dict_of_core_words_in_a.setdefault( term_cui, [] )

            #
            dict_of_core_words_in_a[term_cui].append( core_word_info )

            # [ADD] 增加父子级结点比对的能力
            # 如果 ent_a 具有子结点，将子结点设为 key, 加入到 dict_of_core_words_in_a
            # 发热 比对 高热
            if term_cui in self.children_terms_from_PubSyns:
                list_of_children_terms = self.children_terms_from_PubSyns[term_cui]
                # print( " list_of_children_terms ", list_of_children_terms )
                #
                for child_term_cui in list_of_children_terms:
                    if child_term_cui not in dict_of_core_words_in_a:
                        dict_of_core_words_in_a.setdefault( child_term_cui, [] )

                    # 加入 ent_a 子结点的 cui
                    dict_of_core_words_in_a[child_term_cui].append( core_word_info )

            # [ADD]
            # 高热 比对 发热
            if term_cui in self.parent_terms_from_PubSyns:
                list_of_parent_terms = self.parent_terms_from_PubSyns[term_cui]
                #
                for parent_term_cui in list_of_parent_terms:
                    if parent_term_cui not in dict_of_core_words_in_a:
                        dict_of_core_words_in_a.setdefault( parent_term_cui, [] )

                    # 加入 ent_a 父结点的 cui
                    dict_of_core_words_in_a[parent_term_cui].append( core_word_info )   

            # [ADD] 加入 ent_a 的核心词部分
            # 术语不可分割的核心部分，默认为自身
            term_core_part = term_str
            # 
            if term_str in self.dict_of_term_splits_info:
                # 严重头晕  ['头晕', '严重']
                term_parts = self.dict_of_term_splits_info[term_str]
                term_core_part = term_parts[0]
                # 对 term_core_part 尝试进行标准化
                # 利用 flashtext 关键词抽取器的特性
                # clean_name : ent_info = std_term_str + '||' + ent_tui + '||' + ent_sty + '||' + core_type
                if term_core_part in self.coreword_processor_cn:
                    term_clean_name = self.coreword_processor_cn[term_core_part]
                    term_core_part = term_clean_name.split('||')[0]


            if term_core_part not in dict_of_core_words_in_a:
                dict_of_core_words_in_a.setdefault( term_core_part, [] )

            # 
            dict_of_core_words_in_a[term_core_part].append( core_word_info )





        # 扫描 text_str_b 中的核心词
        # ('痰多||T184||Sign or Symptom||表型核心词'
        core_words_found_in_b = self.coreword_processor_cn.extract_keywords( text_str_b, span_info=True )
        # 调用 filter 函数
        core_words_found_in_b = filter_keywords_found( text_str_b, core_words_found_in_b )
        # print("core_words_found_in_b ", text_str_b, core_words_found_in_b)

        #
        dict_of_core_words_in_b = {}

        #
        for core_word_info in core_words_found_in_b:
            #
            span_info, span_start, span_end = core_word_info

            # info
            term_cui, term_tui, term_sty, core_type = span_info.split("||")

            # 
            term_str = text_str_b[span_start:span_end+1]            

            # 
            if term_cui not in dict_of_core_words_in_b:
                dict_of_core_words_in_b.setdefault( term_cui, [] )

            # 
            dict_of_core_words_in_b[term_cui].append( core_word_info )

            #
            # [ADD] 加入 ent_b 的核心词部分
            # 术语的核心部分的比对能力
            # 术语不可分割的核心部分，默认为自身
            term_core_part = term_str
            # 
            if term_str in self.dict_of_term_splits_info:
                # 急性喉炎  ['喉炎', '急性']
                term_parts = self.dict_of_term_splits_info[term_str]
                term_core_part = term_parts[0]
                # print("term split info before std ", term_str, term_core_part)
                # 对 term_core_part 尝试进行标准化
                # 利用 flashtext 关键词抽取器的特性
                # clean_name : ent_info = std_term_str + '||' + ent_tui + '||' + ent_sty + '||' + core_type
                if term_core_part in self.coreword_processor_cn:
                    term_clean_name = self.coreword_processor_cn[term_core_part]
                    term_core_part = term_clean_name.split('||')[0]
                    # print("term split info after std ", term_str, term_core_part)


            if term_core_part not in dict_of_core_words_in_b:
                dict_of_core_words_in_b.setdefault( term_core_part, [] )

            # 
            dict_of_core_words_in_b[term_core_part].append( core_word_info )



        # 获取 overlapped 核心词
        # 主要先考虑同义词 [简单比对，先考虑同意词就可以了]
        mapped_term_cuis = set( dict_of_core_words_in_a.keys() ) & set( dict_of_core_words_in_b.keys() )
        #  mapped_term_cuis  {'白细胞(WBC)', '发热', '高热'}
        # print(" mapped_term_parts ", mapped_term_cuis) 
        # print( "mapped_term_cuis ", dict_of_core_words_in_a.keys() , dict_of_core_words_in_b.keys() )
        # print( "dict_of_core_words_in_a ", dict_of_core_words_in_a )
        # print( "dict_of_core_words_in_b ", dict_of_core_words_in_b )


        # text_a 中需要 highlight 的 keywords
        for mapped_term_cui in mapped_term_cuis:
            # 获取 text_a 中的 term_name
            for core_word_info_a in dict_of_core_words_in_a[mapped_term_cui]:
                #
                span_info, span_start, span_end = core_word_info_a       
                # ent_a 在文本中的描述
                core_term_of_ent_a = text_str_a[span_start:span_end+1]

                #
                if core_term_of_ent_a not in dict_of_align_info:
                    dict_of_align_info.setdefault( core_term_of_ent_a, [] )

                # 将 b 中与 a 对应的词加入到 dict_of_align_info[term_in_a]
                for core_word_info_b in dict_of_core_words_in_b[mapped_term_cui]:
                    #
                    span_info, span_start, span_end = core_word_info_b
                    # ent_b 在文本中的描述
                    core_term_of_ent_b = text_str_b[span_start:span_end+1]

                    #
                    if core_term_of_ent_b not in dict_of_align_info[core_term_of_ent_a]:
                        dict_of_align_info[core_term_of_ent_a].append( core_term_of_ent_b )

        #
        return dict_of_align_info





if __name__ == '__main__':
    # 
    start_time = time.perf_counter()

    #
    print("start creating MySSUExtractor instance")
    MySSUExtractor = SSUExtractor() 
    print( "Time used in creating MySSUExtractor instance : ", str( round( time.perf_counter() - start_time, 3) ) + ' s' ) 

    # 
    print("\n\nstart calling MySSUExtractor functions")
    start_time = time.perf_counter()


    # 【比对函数】 高级文本比对 (SSU)
    print( "\n\n*******************************************************")
    print( "[高级文本比对函数] get_ssu_alignment_btw_texts() [输出] ")
    print( "*******************************************************")
    text_str_a = '甲硝唑'
    text_str_b = "重度腹痛"
    align_info_dict = MySSUExtractor.get_ssu_alignment_btw_texts( text_str_a, text_str_b, "user001" )
    print( json.dumps(align_info_dict, ensure_ascii=False, indent=4) )
