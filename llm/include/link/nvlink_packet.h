#pragma once

#include <cstdint>
#include <vector>

#include "systemc.h"

// namespace nvlink{
class nvHeaderMsg{
public:
    int opcode; // 4位，代表读/写/原子操作等
    int tag_id;
    int packet_length; //后续flit
};

class nvMsg{
public: 
    int crc; // 127 - 103
    nvHeaderMsg header; // 102 - 20
    int dl_hdr;
    sc_bv<128> ae; //Address Extension 128b，里面有32b是地址
    sc_bv<128> be; //byte Enable
    std::vector<sc_bv<128>> payload;
    
    nvMsg();
    ~nvMsg();

};

// }

// #pragma once

// #include <cstdint>
// #include <vector>
// #include <iostream>
// #include <stdexcept>

// namespace nvlink {
//     enum class CmdType : uint8_t { READ, WRITE, ACK };

//     struct Packet {
//         CmdType cmd;
//         uint64_t addr;
//         std::vector<uint8_t> payload;

//         // Serialize packet to a byte vector
//         std::vector<uint8_t> encode() const;
//         // Deserialize packet from a byte vector
//         static Packet decode(const std::vector<uint8_t>& data);
//         // Print packet info for debugging
//         void print() const;
//     };
// } // namespace nvlink
