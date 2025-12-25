#pragma once
#include "link/instr/chip_instr.h"
#include "link/instr/recv_global_mem.h"
#include "link/instr/print_msg.h"
#include "link/instr/wait_event.h"

chip_instr_base* new_chip_prim(string type){
    chip_instr_base* instr = nullptr;
    
    if(type == "Recv_global_memory"){
        instr = new Recv_global_mem();
    }else if(type == "Print_msg"){
        instr = new Print_msg();
    }
    else if(type == "Wait_event"){
        instr = new Wait_event();
    }
    // else if(type == "Send_global_memory"){

    // }

    else{
        std::cout << "Chip Prim Not Implemented: " << type << std::endl;
        assert(0 && "Chip Prim Not Implemented");
    }

    g_chip_prim_stash.push_back(instr);

    return instr;
}