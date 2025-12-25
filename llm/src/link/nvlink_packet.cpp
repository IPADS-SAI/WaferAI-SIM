// #include "nvlink_packet.h"


// // namespace nvlink{

// nvMsg::nvMsg(){
    
// }

// nvMsg::~nvMsg(){
    
// }

// // }



// // #include "nvlink_packet.h"

// // namespace nvlink {

// // std::vector<uint8_t> Packet::encode() const {
// //     std::vector<uint8_t> buf;
// //     buf.reserve(1 + sizeof(addr) + payload.size());
// //     buf.push_back(static_cast<uint8_t>(cmd));
// //     // Serialize address in little endian
// //     for (size_t i = 0; i < sizeof(addr); ++i) {
// //         buf.push_back(static_cast<uint8_t>((addr >> (8 * i)) & 0xFF));
// //     }
// //     // Append payload
// //     buf.insert(buf.end(), payload.begin(), payload.end());
// //     return buf;
// // }

// // Packet Packet::decode(const std::vector<uint8_t>& data) {
// //     Packet pkt;
// //     size_t min_size = 1 + sizeof(pkt.addr);
// //     if (data.size() < min_size) {
// //         throw std::runtime_error("Invalid packet size: too small");
// //     }
// //     pkt.cmd = static_cast<CmdType>(data[0]);
// //     pkt.addr = 0;
// //     for (size_t i = 0; i < sizeof(pkt.addr); ++i) {
// //         pkt.addr |= static_cast<uint64_t>(data[1 + i]) << (8 * i);
// //     }
// //     pkt.payload.assign(data.begin() + 1 + sizeof(pkt.addr), data.end());
// //     return pkt;
// // }

// // void Packet::print() const {
// //     std::cout << "Cmd: " << static_cast<int>(cmd)
// //               << ", Addr: 0x" << std::hex << addr << std::dec
// //               << ", Payload size: " << payload.size() << std::endl;
// // }

// // } // namespace nvlink
