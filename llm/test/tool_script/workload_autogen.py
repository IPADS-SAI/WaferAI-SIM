import os
import json
import argparse
import re
global input_vars


def init_vars(input_vars):
    input_vars["C"] = int(input_vars['DH'] * input_vars['NH'])
    input_vars["R"] = int(input_vars['NH'] / input_vars['KVH'])
    input_vars["P"] = int(input_vars['HS'])
    input_vars["J"] = int(input_vars['IS'])
    input_vars["chunk"] = 1
    input_vars["loop"] = int(input_vars["avg_output"] + 1)

    input_vars['G'] = int(input_vars["C"] + 2 * input_vars["C"] / input_vars["R"])
    add_vars(input_vars,  "BTC")


def add_vars(input_vars, keys):
    if keys not in input_vars.keys():
        [num, den], type = find_const(keys)
        if type in input_vars.keys():
            value = int(num * input_vars[type] / den)
            input_vars[keys] = value
        else:
            if "/" in keys:
                keys_first, keys_num = keys.split("/")
            else:
                keys_first = keys
                keys_num = 1
            value = 1
            for key in keys_first:
                if key.isdigit():
                    value *= int(key)
                elif key.isalpha():
                    value *=  input_vars[key]

            value =  int(value / int(keys_num))
            if value == 0:
                value = 1
            input_vars[keys] = value


def process_source(input_vars):
    mn_num = input_vars['mn']
    k_num = input_vars['k']

    source = []
    for dp_index in range(input_vars['dp']):
        for k_index in range(k_num):
            for mn_index in range(mn_num):
                if mn_num == 1 :
                    size =f"BTP"
                else:
                    size =f"BTP/{mn_num}"
                if mn_num == 1:
                    dest_id = dp_index * k_num * mn_num * input_vars['pp'] + mn_index * input_vars['mn'] + k_index
                else:
                    dest_id = dp_index * k_num * mn_num * input_vars['pp'] + k_index * input_vars['mn'] + mn_index

                # print(dest_id)
                source.append({"dest": dest_id, "size": size})

                add_vars(input_vars, size)

    return source


def find_const_Dasda(word):
    # print(f"finding word {word}")
    # 修改了一下分词逻辑，应该和之前没有太大区别

    parts = word.split("/")

    head = parts[0]

    rest = parts[1:]
    # 匹配：开头数字 + 后缀
    m = re.match(r"^(\d+)(.*)$", head)

    if m:
        num_part = int(m.group(1))
        word_type = m.group(2)
    else:
        num_part = 1
        word_type = head

    # 处理后面的数字
    if len(rest) == 0:
        rest = 1

    word_all = [num_part, rest]

    # print(f"word_all {word_all}, word type {word_type}")
    return word_all, word_type


def find_const(word):
    word_all = word.split("/")
    num = ""
    word_type = ""
    for index, strs in enumerate(word_all[0]):
        if strs.isdigit():
           num += strs
        else:
            word_type = word_all[0][index:]
            break
    word_all[0] = num
    if word_all[0] == "":
        word_all[0] = "1"
    if len(word_all) == 1:
        word_all.append("1")
    for i in range(2):
        word_all[i] = int(word_all[i])

    return word_all, word_type


def cal_size(word1, word2=None , word3=None, mul_num=None, div_num=None):
    [word1_nun, word1_den], word1_type = find_const(word1)
    if word2 is not None:
        [word2_num, word2_den], word2_type = find_const(word2)
    else:
        [word2_num, word2_den], word2_type = [1,1], ""
    if word3 is not None:
        [word3_num, word3_den], word3_type = find_const(word3)
    else:
        [word3_num, word3_den], word3_type = [1,1], ""

    num = word1_nun * word2_num * word3_num
    den = word1_den * word2_den * word3_den
    if mul_num is not None:
        if mul_num % den == 0:
            num = num * mul_num / den
            den = 1
        elif den % mul_num == 0:
            den = int(den / mul_num)
        else:
            num = num * mul_num

    if div_num is not None:
        if num % div_num == 0:
            num = num / div_num
        elif div_num % num == 0:
            den = int(div_num / num) * den
            num = 1
        else:
            den = den * div_num

    if num == den:
        return_word = f"{word1_type}{word2_type}{word3_type}"
    elif num % den == 0:
        return_word = f"{int(num/den)}{word1_type}{word2_type}{word3_type}"
    elif den % num == 0:
        return_word = f"{word1_type}{word2_type}{word3_type}/{int(den / num)}"
    else:
        return_word = f"{num}{word1_type}{word2_type}{word3_type}/{den}"

    return return_word


gpt = ["layernorm", "matmul", "attention", "matmul", "residual", "layernorm", "matmul", "gelu", "matmul", "residual"]
qwen = ["rmsnorm", "matmul_rope", "attention", "matmul", "residual", "rmsnorm", "matmul×2", "swiglu", "matmul", "residual"]
llama = ["rmsnorm", "matmul_rope", "attention", "matmul", "residual", "rmsnorm", "matmul×2", "swiglu", "matmul", "residual"]




def produce_recv_cast_tag(recv_id, core_id, cast_id, base_tag=64):
    if cast_id is not None:
        recv_tag = core_id + base_tag
        cast_tag = cast_id + base_tag
        return  recv_tag, cast_tag
    else:
        return None, None


def split_prims(prims, id):

    primslist = []
    first_index = 0
    for index, prim in enumerate(prims):
        # if id == 0:
        #     print(prim['type'], prim['sram_address'])
        if prim["type"] == "parse_input" and index - first_index !=0:
            primslist.append(prims[first_index:index])
            first_index = index
        if prim['type'] == 'parse_output' and index != len(prims) - 1:
            primslist.append(prims[first_index:index + 1])
            first_index = index + 1

    primslist.append(prims[first_index:])

    done_worklist = []
    for prim_list in primslist:
        work_prim = []
        one_work = {
            'recv_cnt':0,
        }
        for prim in prim_list:
            if "recv_cnt" in prim.keys():
                one_work["recv_cnt"] = prim["recv_cnt"]
                prim.pop("recv_cnt")
                if "recv_tag" in prim.keys():
                    one_work["recv_tag"] = prim["recv_tag"]
                    prim.pop("recv_tag")
            if "cast" in prim.keys():
                one_work["cast"] = prim["cast"]
                prim.pop("cast")
            work_prim.append(prim)
        if "cast" not in one_work:
            one_work["cast"] = []
        one_work["prims"] = work_prim

        done_worklist.append(one_work)
        # break

    return done_worklist


def add_rope(B, T, C, NH, R, sram_indata, rp_num, prims_list):
    rp_prim = {
        "type": "rope_forward_pd",
        "B": B,
        "T": T,
        "C": C,
        "NH": NH,
        "R": R,
        "sram_address": {
            "indata": sram_indata,
            "outdata": f"rope{rp_num}_out"
        },
        "dram_address": {
            "data": f"rope{rp_num}_data"
        }
    }
    prims_list.append(rp_prim)
    sram_indata = f"rope{rp_num}_out"
    rp_num += 1
    return prims_list, sram_indata, rp_num


def process_one_work_mnk(input_vars, operation, core_layer, core_id, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id, last_layer):
    global oc, mm_type
    prims_list = []

    ln_num = 1
    mm_num = 1
    att_num = 1
    res_num = 1
    gelu_num = 1
    rp_num = 1
    swiglu_num = 1

    B = cal_size("B", div_num=input_vars['dp'])
    add_vars(input_vars, B)

    if input_vars['k'] != 1:
        NH = f"NH/{input_vars['k']}"
    elif input_vars['mn'] != 1:
        NH = f"NH/{input_vars['mn']}"
    else:
        NH = f"NH/{input_vars['mn']}"
    NH = cal_size(NH)
    add_vars(input_vars, NH)

    mn_recv_tag, mn_cast_tag = produce_recv_cast_tag(mn_recv_id, core_id, mn_cast_id, base_tag=1000)
    k_recv_tag, k_cast_tag = produce_recv_cast_tag(k_recv_id, core_id, k_cast_id, base_tag=2000)

    if input_vars['mn'] != 1 and input_vars['k'] != 1:
        id_den = input_vars['mn']
    else:
        id_den = 1

    ic = "P"
    sram_indata = "_input_label"
    res_start = sram_indata
    res_end = res_start

    if input_vars['mn'] != 1 or input_vars['k'] != 1:
        size = f"BT{ic}/{input_vars['mn']}"
        size = cal_size(size)
        add_vars(input_vars, size)
        prim = {
            "type": "parse_input",
            "size": size,
            "sram_address": {
                "indata": "layernorm1_in",
                "outdata": "layernorm1_in"
            },
            "recv_cnt":1
        }
        prims_list.append(prim)

        sram_indata = "_layernorm1_in"
        res_start = "layernorm1_in"


    for layer_index in range(core_layer):
        for num, operates in enumerate(operation):
            add_vars(input_vars, ic)
            if "norm" in operates:
                if input_vars['mn'] != 1:
                    T = f"T/{input_vars['mn']}"
                else:
                    T = "T"
                add_vars(input_vars, T)

                ln_type = "Layernorm_f"
                if operates == "rmsnorm":
                    ln_type = "rmsnorm_forward"
                ln_prim = {
                    "type": ln_type,
                    "B": B,
                    "T": T,
                    "C": "P",
                    "sram_address": {
                        "indata": sram_indata,
                        "outdata": f"layernorm{ln_num}_out"
                    },
                    "dram_address": {
                        "data": f"layernorm{ln_num}_data"
                    }
                }
                if input_vars['mn'] * input_vars['k'] == 1:
                    ln_prim['recv_cnt'] = 1
                sram_indata = f"layernorm{ln_num}_out"
                prims_list += [ln_prim]

                if input_vars['k'] != 1:
                    sd_indata = cal_size(B, T, f"P")

                    sd_outdata = cal_size(B, T, f"P/{input_vars['k']}")
                    add_vars(input_vars, sd_indata)
                    add_vars(input_vars, sd_outdata)

                    sd_prim = {
                        "type": "switch_data",
                        "IN": sd_indata,
                        "OUT": sd_outdata,
                        "sram_address": {
                            "indata": f"layernorm{ln_num}_out",
                            "outdata": f"switch_layernorm{ln_num}_out"
                        }
                    }
                    sram_indata = f"switch_layernorm{ln_num}_out"
                    prims_list += [sd_prim]

                oc = "P"
                ln_num += 1
            elif "matmul" in operates:
                if num == 1:
                    oc = "G" if "rope" in operates else "3C"
                    mm_type = "matmul_forward_pd"
                elif num == 3:
                    oc = "P"
                    mm_type = "Matmul_f"
                elif num == 6:
                    oc = f"J"
                    mm_type = "Matmul_f"
                elif num == 8:
                    oc = "P"
                    mm_type = "Matmul_f"
                mm_time = operates.split("×")[1:]
                if len(mm_time) == 0:
                    mm_time = 1
                else:
                    mm_time = int(mm_time[0])

                T = cal_size(f"T/{input_vars['mn']}")
                add_vars(input_vars, T)

                mm_ic = cal_size(f"{ic}/{input_vars['k']}")
                add_vars(input_vars, mm_ic)

                mm_oc = cal_size(f"{oc}/{input_vars['mn']}")
                add_vars(input_vars, mm_oc)
                swiglu_indata = ""
                for mm_index in range(mm_time):
                    # mn
                    mm_split_num = 1
                    mm_indata = sram_indata
                    if input_vars['mn'] != 1:
                        mm_indata = f"_{sram_indata}"

                    if input_vars['mn'] != 1:
                        mm_outdata = f"matmul{mm_num}_{mm_split_num}_out"
                    else:
                        mm_outdata = f"matmul{mm_num}_out"
                    mm_prim = {
                        "type": mm_type,
                        "B": B,
                        "T": T,
                        "C": mm_ic,
                        "OC": mm_oc,
                        "sram_address": {
                            "indata": mm_indata,
                            "outdata": mm_outdata
                        },
                        "dram_address": {
                            "data": f"matmul{mm_num}_data"
                        }
                    }
                    if mm_type == "matmul_forward_pd":
                        mm_prim_pd = {
                            "R": "R",
                            "chunk": "chunk",
                            "job_type": 2
                        }
                        mm_prim = mm_prim | mm_prim_pd
                    prims_list.append(mm_prim)

                    res_end = mm_outdata
                    mn_merge_indata = f"matmul{mm_num}_{mm_split_num}_out"

                    if "rope" in operates:
                        prims_list, rope_outdata, rp_num = add_rope(B, T, mm_oc, NH, "R", mm_outdata, rp_num, prims_list)
                        mn_merge_indata = rope_outdata

                    for split_mn_index in range(input_vars['mn']-1):
                        inout_size = cal_size(mm_ic, mm_oc)
                        add_vars(input_vars, inout_size)

                        out_prim = {
                            "type": "parse_output",
                            "size": inout_size,
                            "sram_address": {
                                "indata": f"eternal_matmul{mm_num}_{mm_split_num}_w",
                                "outdata": f"DEL_eternal_matmul{mm_num}_{mm_split_num}_w"
                            },
                            "cast":[{"dest": mn_cast_id,
                                     "tag": mn_cast_tag}]
                        }
                        in_prim = {
                            "type": "parse_input",
                            "size": inout_size,
                            "sram_address": {
                                "indata": f"eternal_matmul{mm_num}_{mm_split_num+1}_w",
                                "outdata": f"eternal_matmul{mm_num}_{mm_split_num+1}_w"
                            },
                            "recv_cnt": 1,
                            "recv_tag": mn_recv_tag,
                        }
                        if mm_split_num < input_vars['mn'] - 1:
                            mm_indata = mm_indata
                        else:
                            if mm_index < mm_time - 1:
                                mm_indata = mm_indata
                            else:
                                mm_indata = mm_indata[1:]

                        mm_prim = {
                            "type": mm_type,
                            "B": B,
                            "T": T,
                            "C": mm_ic,
                            "OC": mm_oc,
                            "R": "R",
                            "chunk": "chunk",
                            "job_type": 2,
                            "sram_address": {
                                "indata": mm_indata,
                                "outdata": f"matmul{mm_num}_{mm_split_num + 1}_out"
                            },
                            "dram_address": {
                                "data": f"matmul{mm_num}_data"
                            }
                        }

                        prims_list += [out_prim, in_prim, mm_prim]  if core_id  % 2 == 0 else [in_prim, out_prim, mm_prim]

                        if "rope" in operates:
                            rope_indata = f"matmul{mm_num}_{mm_split_num + 1}_out"
                            prims_list, rope_outdata, rp_num = add_rope(B, T, mm_oc, NH, "R", rope_indata, rp_num, prims_list)
                            mn_merge_indata += " " + rope_outdata
                        else:
                            mn_merge_indata += f" matmul{mm_num}_{mm_split_num + 1}_out"

                        mm_split_num += 1

                    mmm_ic = mm_oc
                    if input_vars['mn'] != 1:
                        mmm_prim = {
                            "type": "Merge_matmul",
                            "B": B,
                            "T": T,
                            "C": mmm_ic,
                            "dim": 1,
                            "slice": input_vars['mn'],
                            "sram_address": {
                                "indata": mn_merge_indata,
                                "outdata": f"matmul{mm_num}_out"
                            }
                        }
                        prims_list += [mmm_prim]
                        mn_mmm_outdata = f"matmul{mm_num}_out"
                        res_end = mn_mmm_outdata


                    # k
                    inout_size = cal_size(B, T, mmm_ic)
                    add_vars(input_vars, inout_size)
                    k_merge_indata = f"matmul{mm_num}_out"
                    if input_vars["k"] != 1:
                        if input_vars["mn"] != 1:
                            sd_in = cal_size(B, f"T/{input_vars['mn']}", mmm_ic, mul_num=input_vars['mn'])
                            sd_out = cal_size(B, f"T/{input_vars['mn']}", mmm_ic, mul_num=input_vars['mn'], div_num=input_vars['k'])
                            sd_indata = mn_mmm_outdata
                            if num == 1 or num == 6:
                                sd_prim = {
                                    "type": "switch_data",
                                    "IN": sd_in,
                                    "OUT": sd_out,
                                    "sram_address": {
                                        "indata": sd_indata,
                                        "outdata": f"k_matmul{mm_num}_1_out"
                                    }
                                }
                                prims_list.append(sd_prim)

                                inout_size = sd_out
                                k_merge_indata = f"k_matmul{mm_num}_1_out"
                                mmm_oc = mmm_ic
                            else:
                                k_merge_indata = f"matmul{mm_num}_out"
                                mmm_oc = cal_size(mmm_ic,mul_num=input_vars['mn'])
                                inout_size = sd_in
                        else:
                            sd_in = cal_size(B, "T", mm_oc)
                            sd_out = cal_size(B, "T", mm_oc, div_num=input_vars['k'])
                            if num == 1 or num == 6:
                                sd_indata = mm_outdata
                            else:
                                sd_indata = "_" + mm_outdata
                            sd_prim = {
                                "type": "switch_data",
                                "IN": sd_in,
                                "OUT": sd_out,
                                "sram_address": {
                                    "indata": sd_indata,
                                    "outdata": f"k_matmul{mm_num}_1_out"
                                }
                            }
                            prims_list.append(sd_prim)
                            inout_size = sd_out
                            mmm_oc = mmm_ic
                            k_merge_indata = f"k_matmul{mm_num}_1_out"
                        add_vars(input_vars, sd_in)
                        add_vars(input_vars, sd_out)

                    if input_vars['k'] != 1:
                        k_merge_num = 1
                        for k_index in range(input_vars['k']-1):
                            if input_vars['mn'] == 1:
                                inout_indata = f"k_matmul{mm_num}_{k_merge_num}_out"
                                inout_outdata = f"k_matmul{mm_num}_{k_merge_num}_out"
                            else:
                                if num == 1 or num == 6:
                                    inout_indata = "_" + f"k_matmul{mm_num}_{k_merge_num}_out"
                                    inout_outdata = f"k_matmul{mm_num}_{k_merge_num}_out"
                                else:
                                    inout_indata = f"matmul{mm_num}_out"
                                    inout_outdata = inout_indata

                            out_prim = {
                                "type": "parse_output",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": inout_indata,
                                    "outdata": inout_outdata
                                },
                                "cast": [{
                                    "dest": k_cast_id,
                                    "tag": k_cast_tag
                                }],
                            }
                            in_prim = {
                                "type": "parse_input",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": f"k_matmul{mm_num}_{k_merge_num+1}_out",
                                    "outdata": f"k_matmul{mm_num}_{k_merge_num+1}_out"
                                },
                                "recv_cnt": 1,
                                "recv_tag": k_recv_tag,
                            }
                            k_merge_indata += f" k_matmul{mm_num}_{k_merge_num+1}_out"
                            prims_list += [out_prim, in_prim] if int(core_id / id_den) % 2 == 0 else [in_prim, out_prim]


                            if input_vars['mn'] != 1:
                                if num == 1 or num == 6:
                                    mmm_oc = mm_oc
                                else:
                                    mmm_oc = cal_size(mm_oc, mul_num=input_vars['mn'])
                                    res_end =  f"k_matmul{mm_num}_out"
                            else:
                                mmm_oc = cal_size(mm_oc, div_num=input_vars['k'])

                            add_vars(input_vars,mmm_oc)
                            if k_index == input_vars['k']-2:
                                T = cal_size(f"T/{input_vars['mn']}")
                                mmm_prim = {
                                    "type": "Merge_matmul",
                                    "B": B,
                                    "T": T,
                                    "C": mmm_oc,
                                    "dim": 2,
                                    "slice": input_vars['k'],
                                    "sram_address": {
                                        "indata": k_merge_indata,
                                        "outdata": f"k_matmul{mm_num}_out"
                                    }
                                }
                                prims_list.append(mmm_prim)
                                k_mmm_outdata = f"k_matmul{mm_num}_out"
                            k_merge_num += 1


                    if input_vars['k'] != 1 and input_vars['mn'] == 1:
                        for k_index in range(input_vars['k']-1):
                            if input_vars['mn'] == 1:
                                inout_indata = f"k_matmul{mm_num}_out"
                                inout_outdata = f"k_matmul{mm_num}_out"
                            else:
                                if num == 1 or num == 6:
                                    inout_indata = "" + f"k_matmul{mm_num}_out"
                                    inout_outdata = f"k_matmul{mm_num}_out"
                                else:
                                    inout_indata = f"matmul{mm_num}_out"
                                    inout_outdata = inout_indata

                            out_prim = {
                                "type": "parse_output",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": inout_indata,
                                    "outdata": inout_outdata
                                },
                                "cast": [{
                                    "dest": k_cast_id,
                                    "tag": k_cast_tag
                                }],
                            }
                            in_prim = {
                                "type": "parse_input",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": inout_indata,
                                    "outdata": inout_outdata
                                },
                                "recv_cnt": 1,
                                "recv_tag": k_recv_tag,
                            }
                            prims_list += [out_prim, in_prim] if int(core_id / id_den) % 2 == 0 else [in_prim, out_prim]

                    if mm_time != 1:
                        sram_indata = sram_indata
                    elif input_vars['mn'] == 1 and input_vars['k'] == 1:
                        if "rope" in operates:
                            sram_indata = rope_outdata
                        else:
                            sram_indata = mm_outdata
                    elif input_vars['mn'] != 1 and input_vars['k'] == 1:
                        sram_indata = mn_mmm_outdata
                    elif  input_vars['k'] != 1:
                        sram_indata = k_mmm_outdata

                    if mm_time != 1:
                        if input_vars['mn'] == 1 and input_vars['k'] == 1:
                            swiglu_indata += mm_outdata + " "
                        elif input_vars['mn'] != 1 and input_vars['k'] == 1:
                            swiglu_indata += mn_mmm_outdata + " "
                        elif input_vars['k'] != 1:
                            swiglu_indata += k_mmm_outdata + " "

                    mm_num += 1
            elif "attention" in operates:
                ic_num = ic[:-1]
                if input_vars['mn'] != 1:
                    inout_size = f"{ic_num}BTC/{input_vars['mn'] * input_vars['k']}"

                    add_vars(input_vars, inout_size)
                    for mn_ndex in range(input_vars['mn']-1):
                        out_prim = {
                            "type": "parse_output",
                            "size": inout_size,
                            "sram_address": {
                                "indata": sram_indata,
                                "outdata": sram_indata
                            },
                            "cast": [{
                                "dest": mn_cast_id,
                                "tag": mn_cast_tag
                            }],
                        }
                        in_prim = {
                            "type": "parse_input",
                            "size": inout_size,
                            "sram_address": {
                                "indata": sram_indata,
                                "outdata": sram_indata
                            },
                            "recv_cnt": 1,
                            "recv_tag": mn_recv_tag,
                        }
                        prims_list += [out_prim, in_prim] if core_id % 2 == 0 else [in_prim, out_prim]


                C = cal_size(f"C/{input_vars['mn'] * input_vars['k']}")
                add_vars(input_vars, C)
                att_prim = {
                    "type": "Attention_f_pd",
                    "B": B,
                    "T": "T",
                    "C": C,
                    "NH": NH,
                    "DH": "DH",
                    "R": "R",
                    "job_type": 2,
                    "sram_address": {
                        "indata": sram_indata,
                        "outdata": f"attention{att_num}_out"
                    },
                    "dram_address": {
                        "data": f"attention{att_num}_data",
                        "out": "TODO"
                    }
                }
                prims_list.append(att_prim)
                sram_indata = f"attention{att_num}_out"

                if input_vars['mn'] != 1:
                    inout_size = f"BTC/{input_vars['mn'] * input_vars['k']}"

                    for mn_ndex in range(input_vars['mn']-1):
                        out_prim = {
                            "type": "parse_output",
                            "size": inout_size,
                            "sram_address": {
                                "indata": sram_indata,
                                "outdata": sram_indata
                            },
                            "cast": [{
                                "dest": mn_cast_id,
                                "tag": mn_cast_tag
                            }],
                        }
                        in_prim = {
                            "type": "parse_input",
                            "size": inout_size,
                            "sram_address": {
                                "indata": sram_indata,
                                "outdata": sram_indata
                            },
                            "recv_cnt": 1,
                            "recv_tag": mn_recv_tag,
                        }
                        prims_list += [out_prim, in_prim] if core_id % 2 == 0 else [in_prim, out_prim]

                sram_indata = f"attention{att_num}_out"
                oc = "C"
                att_num += 1
            elif "residual" in operates:
                N = cal_size(B, "T", f"{ic}/{input_vars['mn']}")
                add_vars(input_vars, N)
                res_prim = {
                    "type": "Residual_f",
                    "N": N,
                    "sram_address": {
                        "indata": f"{res_start} {res_end}",
                        "outdata": f"residual{res_num}_out"
                    }
                }

                if num == 9 and layer_index == core_layer-1:
                    if not last_layer:
                        res_prim["cast"] = [dict(dest=core_id + input_vars['k'] * input_vars['mn'])]
                    else:

                        loop_id = core_id - input_vars['k'] * input_vars['mn'] * (input_vars["pp"] - 1)
                        res_prim["cast"] = [{
                                "dest": -1,
                                "loopout": "true"
                            },
                            {
                                "dest": loop_id,
                                "loopout": "false"
                            }]

                prims_list.append(res_prim)
                sram_indata = f"_residual{res_num}_out"
                res_start = f"residual{res_num}_out"
                res_num += 1
            elif "gelu" in operates:
                N = cal_size(B, f"T/{input_vars['mn']}", f"{ic}/{input_vars['k']}")
                add_vars(input_vars, N)
                gelu_prim = {
                    "type": "Gelu_f",
                    "N": N,
                    "sram_address": {
                        "indata": sram_indata,
                        "outdata": f"gelu{gelu_num}_out"
                    },
                    "dram_address": {
                        "data": f"gelu{gelu_num}_data",
                        "out": "TODO"
                    }
                }
                prims_list.append(gelu_prim)
                sram_indata = f"gelu{gelu_num}_out"
                gelu_num += 1
            elif "swiglu" in operates:
                N = cal_size(B, f"T/{input_vars['mn']}", f"{ic}/{input_vars['k']}")
                add_vars(input_vars, N)
                swiglu_prim = {
                    "type": "swiglu_forward",
                    "N": N,
                    "sram_address": {
                        "indata": swiglu_indata[:-1],
                        "outdata": f"swiglu{swiglu_num}_out"
                    },
                    "dram_address": {
                        "input": 0,
                        "data": -1
                    }
                }
                prims_list.append(swiglu_prim)
                sram_indata = f"swiglu{swiglu_num}_out"
                swiglu_num += 1
            ic = oc

    return prims_list


def process_worklist_mnk(input_vars, core_id, core_layer, operation, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id, last_layer):
    pirms = process_one_work_mnk(input_vars, operation, core_layer, core_id, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id, last_layer)
    worklist = split_prims(pirms, core_id)
    return worklist


def layer_adapt_pp(input_vars):
    cores_list = []
    if input_vars["pp"] >= input_vars["L"]:
        cores_list = [1] * input_vars["L"]
    else:
        cores_num = int(input_vars["L"] / input_vars["pp"])
        cores_list = [cores_num] * input_vars["pp"]
        addtional_op_num = input_vars["L"] - cores_num * input_vars["pp"]
        for i in range(addtional_op_num):
            cores_list[i] = cores_list[i] + 1

    return cores_list


def process_cores(input_vars):
    global core_id, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id
    cores_list = layer_adapt_pp(input_vars)
    # print(cores_list)

    decoder = gpt
    if input_vars['model'] == "qwen":
        decoder = qwen

    cores = []
    for dp_index in range(input_vars['dp']):
        for core_num, core_layer in enumerate(cores_list):
            for k_index in range(input_vars["k"]):
                for mn_index in range(input_vars["mn"]):
                    dp_base = dp_index * input_vars['pp'] * input_vars['k'] * input_vars['mn']
                    if input_vars['k'] != 1 and input_vars["mn"] != 1:
                        core_id = dp_base + core_num * input_vars["k"] * input_vars["mn"] + k_index * input_vars["mn"] + mn_index
                        if mn_index < input_vars["mn"] - 1:
                            mn_cast_id = dp_base + core_num * input_vars["k"] * input_vars["mn"] + k_index * input_vars["mn"] + mn_index + 1
                        else:
                            mn_cast_id = dp_base + core_num * input_vars["k"] * input_vars["mn"] + (k_index-1) * input_vars["mn"] + mn_index + 1

                        if mn_index > 0:
                            mn_recv_id = dp_base + core_num * input_vars["k"] * input_vars["mn"] + k_index * input_vars["mn"] + mn_index - 1
                        else:
                            mn_recv_id = dp_base + core_num * input_vars["k"] * input_vars["mn"] + (k_index+1) * input_vars["mn"] + mn_index - 1

                        if k_index < input_vars["k"] - 1 :
                            k_cast_id = dp_base + core_num * input_vars["k"] * input_vars["mn"] + (k_index + 1) * input_vars["mn"] + mn_index
                        else:
                            k_cast_id = dp_base + (core_num - 1) * input_vars["k"] * input_vars["mn"] + (k_index + 1) * input_vars["mn"] + mn_index

                        if k_index > 0:
                            k_recv_id = dp_base + core_num * input_vars["k"] * input_vars["mn"] + (k_index - 1) * input_vars["mn"] + mn_index
                        else:
                            k_recv_id = dp_base + (core_num + 1) * input_vars["k"] * input_vars["mn"] + (k_index - 1) * input_vars["mn"] + mn_index
                    elif input_vars['k'] == 1 and input_vars["mn"] != 1:
                        core_id = core_num * input_vars["mn"] + mn_index
                        mn_cast_id = dp_base + core_num * input_vars["mn"] + mn_index + 1 if mn_index < input_vars["mn"] - 1 else dp_base + core_num * input_vars["mn"]
                        mn_recv_id = dp_base + core_num * input_vars["mn"] + mn_index - 1 if mn_index > 0 else dp_base + core_num * input_vars["mn"] + 3
                        k_cast_id = None
                        k_recv_id = None
                    elif input_vars['k'] != 1 and input_vars["mn"] == 1:
                        core_id = core_num * input_vars["k"] + k_index
                        mn_cast_id = None
                        mn_recv_id = None
                        k_cast_id = dp_base + core_num * input_vars["k"] + k_index + 1 if k_index < input_vars["k"] - 1 else dp_base + core_num * input_vars["k"]
                        k_recv_id = dp_base + core_num * input_vars["k"] + k_index - 1 if k_index > 0 else dp_base + core_num * input_vars["k"] + 3
                    else:
                        core_id = dp_index * input_vars['pp'] + core_num
                        mn_recv_id = None
                        mn_cast_id = None
                        k_recv_id = None
                        k_cast_id = None

                    # print(
                    #     f"mn_recv_id: {mn_recv_id}, "
                    #     f"k_recv_id: {k_recv_id}, "
                    #     f"core_id: {core_id}, "
                    #     f"mn_cast_id: {mn_cast_id}, "
                    #     f"k_cast_id: {k_cast_id}"
                    # )

                    core = {"id": core_id,
                            "loop": "loop",
                            "worklist": process_worklist_mnk(input_vars, core_id, core_layer, decoder, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id, core_num == len(cores_list) - 1)}
                    cores.append(core)
    return cores


def process_chips(input_vars):
    chips = {"chip_id":0, "cores": process_cores(input_vars)}
    return [chips]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="dir where stores output traces", default="./test", required=False)
    parser.add_argument("--output_name", type=str, help="name of output traces", default="tp_test", required=False)
    parser.add_argument("--B", type=int, help="Batch_size", default=1, required=False)
    parser.add_argument("--T", type=int, help="seq length", default=54, required=False)
    parser.add_argument("--DH", type=int, help="dimension of head", default=128, required=False)
    parser.add_argument("--NH", type=int, help="number of heads", default=32, required=False)
    parser.add_argument("--KVH", type=int, help="KV heads", default=8, required=False)
    parser.add_argument("--HS", type=int, help="hidden size", default=2560, required=False)
    parser.add_argument("--L", type=int, help="transformer layers", default=1, required=False)
    parser.add_argument("--pp", type=int, help="pipeline parallel", default=1, required=False)
    parser.add_argument("--dp", type=int, help="dataset parallel", default=1, required=False)
    parser.add_argument("--tp", type=str, help="tensor parallel, mn_k", default="1_1", required=False)
    parser.add_argument("--IS", type=int, help="intermediate size", default=9728, required=False)
    parser.add_argument("--avg_output", type=int, help="average output tokens", default=10, required=False)
    parser.add_argument("--model", type=str, help="gpt or qwen", default="gpt", required=False)

    input_vars = vars(parser.parse_args())
    file_path = input_vars["output_dir"]
    file_name = input_vars["output_name"]
    input_vars.pop("output_dir")
    input_vars.pop("output_name")

    varitations = {
        "layernorm1_data": 0,
        "split_matmul1_out": 0,
        "matmul1_data": 0,
        "attention1_data": 0,
        "matmul2_data": 0,
        "matmul2_out": 0,
        "merge_matmul1_in": 0,
        "residual1_out": 0,
        "matmul3_data": 0,
        "matmul4_data": 0,
        "matmul4_out": 0,
        "layernorm2_data": 0,
        "split_matmul2_out": 0,
        "residual2_out": 0
    }
    input_vars = input_vars | varitations
    init_vars(input_vars)

    file_name = f"{input_vars['model']}_{file_name}_{input_vars['tp']}"
    input_vars['tp'] = input_vars['tp'].split("_")
    input_vars['mn'], input_vars['k'] = [int(i) for i in input_vars['tp']]
    configs = {"vars": input_vars,
               "pipeline": 1,
               "source": process_source(input_vars),
               "chips": process_chips(input_vars)
               }

    input_vars.pop('tp')
    input_vars.pop('model')

    # print(input_vars['tp'], input_vars['k'])
    # print(configs)
    os.makedirs(file_path, exist_ok=True)
    with open(f"{file_path}/{file_name}.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=4)


if __name__ == '__main__':
    main()